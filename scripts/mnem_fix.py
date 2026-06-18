"""
Mnemosyne data quality fix pipeline:
1. Purge garbage consolidated_facts (malformed triples, truncated garbage)
2. Re-embed all working_memory rows missing embeddings
3. Mark unresolved conflicts with a resolution

SAFETY: This script writes 384-dim vectors to the local SQLite memory_embeddings table ONLY.
The Qdrant pipeline uses 768-dim nomic-embed-text. Do NOT add Qdrant writes here.
Run with --dry-run first to preview changes.
"""
import sqlite3, json, os, sys
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DRY_RUN = "--dry-run" in sys.argv

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")
test_vec = model.encode("dimension check", normalize_embeddings=True)
assert len(test_vec) == 384, f"ABORT: Expected 384-dim from {MODEL_NAME}, got {len(test_vec)}. Wrong model loaded?"
print(f"Model dimension: {len(test_vec)} (expected 384 for local SQLite embeddings)")

DBS = {
    "default": os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
    "dama-gotchi": os.path.expanduser("~/.hermes/mnemosyne/data/banks/dama-gotchi/mnemosyne.db"),
}

# Garbage patterns for consolidated_facts (malformed/truncated triples)
GARBAGE_SUBJECTS = [
    "Here", "Note that", "Note that this summary", "Based on", "This",
]
GARBAGE_OBJECTS_PARTIAL = [
    "concise", "based", "ccessed",  # truncated words
]
# Short/meaningless objects (single word stopwords)
STOPWORD_OBJECTS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}

def is_garbage_fact(subject, predicate, obj, confidence):
    if subject.strip() in GARBAGE_SUBJECTS:
        return True
    if obj is not None and str(obj).strip().lower() in STOPWORD_OBJECTS:
        return True
    if obj is not None and str(obj).strip() in GARBAGE_OBJECTS_PARTIAL:
        return True
    # Very short subject/object combo with low confidence = noise
    if len(str(subject)) < 4 and confidence and confidence < 0.5:
        return True
    return False

total_purged = 0
total_embedded = 0

for bank, path in DBS.items():
    if not os.path.exists(path):
        print(f"\n[SKIP] {bank} — path not found")
        continue
    print(f"\n{'='*50}")
    print(f"Processing bank: {bank}")
    db = sqlite3.connect(path)
    cur = db.cursor()

    # --- Step 1: Purge garbage consolidated_facts ---
    cur.execute("SELECT id, subject, predicate, object, confidence FROM consolidated_facts")
    rows = cur.fetchall()
    to_purge = []
    for row in rows:
        id_, subject, predicate, obj, confidence = row
        if is_garbage_fact(subject, predicate, obj, confidence):
            to_purge.append(id_)
            print(f"  PURGE cf id={id_} conf={confidence}: {subject} | {predicate} | {repr(str(obj)[:60])}")

    if to_purge and not DRY_RUN:
        cur.executemany("DELETE FROM consolidated_facts WHERE id = ?", [(id_,) for id_ in to_purge])
        db.commit()
        print(f"  Purged {len(to_purge)} garbage facts from {bank}")
    elif to_purge:
        print(f"  [DRY RUN] Would purge {len(to_purge)} facts")
    else:
        print(f"  No garbage facts to purge")
    total_purged += len(to_purge)

    # --- Step 2: Re-embed missing working_memory rows ---
    cur.execute("""
        SELECT wm.id, wm.content FROM working_memory wm
        LEFT JOIN memory_embeddings me ON me.memory_id = wm.id
        WHERE me.memory_id IS NULL
    """)
    missing = cur.fetchall()
    print(f"\n  Missing embeddings: {len(missing)}")

    for mem_id, content in missing:
        if not content or not content.strip():
            print(f"    SKIP empty content id={mem_id}")
            continue
        print(f"    Embedding id={mem_id}: {repr(content[:60])}")
        if not DRY_RUN:
            vec = model.encode(content, normalize_embeddings=True)
            embedding_json = json.dumps(vec.tolist())
            cur.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model, created_at) VALUES (?, ?, ?, datetime('now'))",
                (mem_id, embedding_json, MODEL_NAME)
            )
        total_embedded += 1

    if not DRY_RUN and missing:
        db.commit()
        print(f"  Embedded {len(missing)} rows in {bank}")

    # --- Step 3: Resolve unresolved conflicts ---
    cur.execute("SELECT id, fact_a_id, fact_b_id, conflict_type FROM conflicts WHERE resolution IS NULL OR resolution = ''")
    conflicts = cur.fetchall()
    print(f"\n  Unresolved conflicts: {len(conflicts)}")
    for conf_id, fa, fb, ctype in conflicts:
        # Fetch both facts
        cur.execute("SELECT subject, predicate, object, confidence FROM consolidated_facts WHERE id IN (?, ?)", (fa, fb))
        facts = cur.fetchall()
        resolution = f"AUTO: Both facts retained; conflict_type={ctype}. Manual review recommended."
        print(f"    Conflict id={conf_id} ({ctype}): {fa} vs {fb}")
        for f in facts:
            print(f"      fact: {f[0]} | {f[1]} | {f[2]} conf={f[3]}")
        if not DRY_RUN:
            cur.execute("UPDATE conflicts SET resolution = ?, resolved_at = datetime('now') WHERE id = ?",
                        (resolution, conf_id))
    if not DRY_RUN and conflicts:
        db.commit()

    db.close()

print(f"\n{'='*50}")
print(f"DONE. Total purged={total_purged} re-embedded={total_embedded}")
if DRY_RUN:
    print("(DRY RUN — no changes written)")
