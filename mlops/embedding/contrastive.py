"""Fine-tune a sentence-transformers embedding model on Loci grounding pairs."""

import argparse
import json
import pathlib
import sys

import numpy as np

try:
    from sentence_transformers import (
        InputExample,
        SentenceTransformer,
        losses,
    )
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
except ImportError:
    sys.exit("pip install sentence-transformers torch")

from sklearn.model_selection import train_test_split

MODEL_CONFIGS = {
    "small": {
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "epochs": 3,
        "trust_remote_code": False,
    },
    "medium": {
        "base": "nomic-ai/nomic-embed-text-v1",
        "epochs": 5,
        # nomic-embed-text-v1 uses custom modeling code not shipped with transformers
        "trust_remote_code": True,
    },
}


def load_dataset(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_examples(rows: list[dict]) -> tuple[list[InputExample], list[float]]:
    """Return (examples, scores) where score is 1.0 for positive, 0.0 for negative."""
    examples = []
    scores = []
    for row in rows:
        score = 1.0 if row["label"] == 1 else 0.0
        examples.append(InputExample(texts=[row["claim"], row["evidence"]], label=score))
        scores.append(score)
    return examples, scores


def baseline_spearman(
    model: SentenceTransformer,
    val_examples: list[InputExample],
    val_scores: list[float],
    out_dir: pathlib.Path,
) -> float:
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name="val-baseline",
        write_csv=False,
    )
    result = evaluator(model, output_path=str(out_dir))
    return result


def print_stats(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    signals = {}
    for r in rows:
        signals[r.get("signal", "unknown")] = signals.get(r.get("signal", "unknown"), 0) + 1
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"Dataset: {len(rows)} rows — pos={pos}, neg={neg}")
    print(f"Signal breakdown: {signals}")
    cos_vals = [r["cos"] for r in rows if "cos" in r]
    if cos_vals:
        print(f"Cosine similarity — mean={np.mean(cos_vals):.3f}, std={np.std(cos_vals):.3f}")


def print_ollama_instructions(model_dir: pathlib.Path) -> None:
    print()
    print("=" * 60)
    print("To use as an Ollama model:")
    print("  # Option A — llama.cpp convert script:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  pip install -r llama.cpp/requirements.txt")
    print(f"  python3 llama.cpp/convert_hf_to_gguf.py {model_dir} \\")
    print(f"      --outfile {model_dir}/model.gguf --outtype q8_0")
    print("  ollama create loci-embed -f Modelfile  # Modelfile: FROM ./model.gguf")
    print()
    print("  # Option B — llama-cpp-python:")
    print("  pip install llama-cpp-python")
    print("  from llama_cpp import llama_model_quantize  # see llama_cpp docs")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for Loci embeddings")
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=pathlib.Path("deep_think_loci/grounding/grounding_dataset.jsonl"),
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "medium"],
        default="small",
        dest="model_size",
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("mlops/embedding/"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Load data and print stats only — no training",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"Dataset not found: {args.dataset}")

    rows = load_dataset(args.dataset)
    print_stats(rows)

    if args.dry_run:
        print("\n--dry-run: exiting before training.")
        return

    cfg = MODEL_CONFIGS[args.model_size]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    model_out_dir = args.out / f"loci-embed-{args.model_size}"

    examples, scores = make_examples(rows)
    labels_for_split = [r["label"] for r in rows]

    train_idx, val_idx = train_test_split(
        range(len(examples)),
        test_size=0.2,
        stratify=labels_for_split,
        random_state=42,
    )
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    val_scores = [scores[i] for i in val_idx]

    print(f"\nTrain={len(train_examples)}, Val={len(val_examples)}")
    print(f"Loading base model: {cfg['base']}")

    model = SentenceTransformer(
        cfg["base"],
        trust_remote_code=cfg["trust_remote_code"],
    )

    print("\nEvaluating untrained baseline on val set...")
    eval_out = args.out / "eval_tmp"
    eval_out.mkdir(parents=True, exist_ok=True)
    baseline_eval = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="val-baseline", write_csv=False
    )
    baseline_score = baseline_eval(model, output_path=str(eval_out))
    print(f"Baseline Spearman (untrained): {baseline_score:.4f}")

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    loss_fn = losses.CosineSimilarityLoss(model)

    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="val", write_csv=False
    )

    warmup_steps = int(len(train_loader) * epochs * 0.10)
    print(f"\nTraining: epochs={epochs}, batch_size={args.batch_size}, warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        evaluator=val_evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(model_out_dir),
        show_progress_bar=True,
    )

    print(f"\nModel saved to: {model_out_dir}")

    print("\nEvaluating fine-tuned model on val set...")
    trained_model = SentenceTransformer(str(model_out_dir), trust_remote_code=cfg["trust_remote_code"])
    final_eval = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="val-final", write_csv=False
    )
    final_score = final_eval(trained_model, output_path=str(eval_out))
    delta = final_score - baseline_score
    print(f"Fine-tuned Spearman: {final_score:.4f}  (delta from baseline: {delta:+.4f})")

    eval_record = {
        "model_size": args.model_size,
        "base_model": cfg["base"],
        "epochs": epochs,
        "batch_size": args.batch_size,
        "train_n": len(train_examples),
        "val_n": len(val_examples),
        "baseline_spearman": round(baseline_score, 6),
        "finetuned_spearman": round(final_score, 6),
        "delta": round(delta, 6),
    }
    eval_path = args.out / "eval.json"
    eval_path.write_text(json.dumps(eval_record, indent=2))
    print(f"Eval results written to: {eval_path}")

    print_ollama_instructions(model_out_dir)


if __name__ == "__main__":
    main()
