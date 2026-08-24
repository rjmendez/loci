# A companion service for Loci — where the passive tier should run

`scripts/loci_groom.py` needs a home. It is unattended, batch-shaped, and wants a
generation tier that is *there* rather than one that is fast. That is a different
requirement from the retrieval path, and the two have been conflated.

This note records why the previous serving attempts went badly — measured on the
live cluster, not recalled — and what shape survives it.

## Why the previous attempts went badly

Not model choice. The substrate.

### The node advertises 32 GPUs and owns roughly one

```
$ kubectl get node desktop-bvrdk4j -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'
32
```

The NVIDIA device plugin under WSL2 advertises a bogus device count. Kubernetes
believes it, so nothing stops it scheduling many `nvidia.com/gpu: 1` pods onto a
node with one physical card. They admit successfully and then contend for the
same device.

### Which produces exactly the failures on record

```
$ kubectl -n dama describe pod vllm-gpu-66d44bdd86-fhth7
Status:   Failed
Reason:   UnexpectedAdmissionError
Message:  Pod was rejected: Allocate failed due to no healthy devices present;
          cannot allocate unhealthy devices nvidia.com/gpu, which is unexpected
```

The device plugin daemonset has restarted **9 times in 52 days**. Each restart is
a window in which devices report unhealthy: running GPU pods die with
`ContainerStatusUnknown` and their replacements are rejected outright. The
ReplicaSet leaves a graveyard behind it — `vllm-gpu` currently shows 1 Running
against 3 dead over 9 days, and `qdrant-mcp-server` the same shape.

Cluster-wide right now: 93 Running, 21 Error, 2 OOMKilled, 2
UnexpectedAdmissionError, 2 ContainerStatusUnknown.

### On a node already promising more than it has

```
Allocated resources:
  cpu     19 (67%) requests    114050m (407%) limits
  memory  55596Mi (57%)        216198Mi (223%) limits
```

`infra/qdrant` — the store everything else depends on — has restarted **32 times
in 93 days**.

**So: a Loci companion service that assumes a GPU pod stays admitted will inherit
all of this.** Every previous attempt did. That is the thing to design around,
and it is not fixed by picking a better model.

## What the work actually needs

The two tiers have been treated as one and they have opposite requirements.

| | retrieval path | passive grooming |
|---|---|---|
| on a user's critical path | yes | no |
| latency budget | milliseconds | hours |
| failure cost | a request degrades | a batch retries later |
| per-call cost tolerance | must be ~0 | a fraction of a cent is fine |
| can it leave the machine | reranker/embedder: no | mostly yes, with a gate |

The reranker and the embedder must stay local — they are called on every
retrieval, and network round-trips would dominate. Nothing about that changes.

The grooming passes are the opposite: idempotent, shadow-first, and resumable by
construction. A pass that fails costs one wasted batch and re-runs on the next
tick. That is precisely the workload that should *not* be pinned to the least
reliable thing in the system.

## The shape: a router, not a server

`mcp/batched_gen.py` already has the right idea — vLLM primary, Ollama fallback,
never raises. Generalise it into a small companion service that owns three tiers
and demotes on health rather than on hope:

| tier | what it is | when it is chosen |
|---|---|---|
| **local batched** | vLLM on the GPU node | healthy, and the batch is large enough to be worth continuous batching |
| **local serial** | Ollama, `keep_alive`-pinned | vLLM absent or the device plugin is mid-flap |
| **remote** | OpenRouter | both local tiers are down, or the pass is explicitly marked remote-eligible |

The service's job is small and worth stating precisely, because scope creep here
is how the previous attempts got heavy:

- resolve a tier per request, from live health, with hysteresis so a flapping
  device plugin does not cause a flapping router
- normalise the model name per tier (`qwen2.5:3b` vs `Qwen/Qwen2.5-3B-Instruct`
  vs `qwen/qwen-2.5-7b-instruct` are the same intent and three different strings —
  this exact mismatch already cost one debugging round in `loci_groom`)
- expose a batch endpoint whose result list is always 1:1 with the prompt list,
  degraded entries included, so callers never have to align by hand
- account for what it spent, per tier, per pass

It should NOT own: the queue (cron owns scheduling), the corpus (JSONL owns it),
promotion (the adjudication tier owns it), or embeddings (they stay local).

## On OpenRouter specifically

The attraction is precise: it is the only tier whose availability is uncorrelated
with this node. When the device plugin flaps, that is exactly when a passive tier
would otherwise stall, so a remote fallback converts an outage into a slightly
more expensive hour.

Three things to decide before wiring it up.

**Redaction is not optional.** The corpus is infrastructure findings. It contains
internal IPs, hostnames, k3s topology, and at least historically a live API key
(#79). A pass that ships finding text to a third party needs either a redaction
gate or an explicit per-pass allow-list. The cleanest rule: **passes declare
`remote_ok`, and it defaults to false.** `recall`'s generated questions and
`codelink`'s disambiguation prompts are low-risk and short; whole-finding
summarisation is not.

**Cost has to be bounded per run, not per call.** A grooming pass over 2,435
findings that silently falls back to a remote tier is a bill, not a degradation.
Give the router a per-invocation budget and let it return `ok: False` when
exhausted — the passes already handle that, since every one of them counts its
rejections.

**Pick the tier per pass, not globally.** `knn_tags` needs no generation at all.
`codelink` needs it only for the ~1,081 ambiguous tokens. `recall` needs one
short question per sampled finding. These have very different remote-cost
profiles and should not share one switch.

## Fixing the substrate anyway

Independent of where generation runs, three things on the node are worth doing
because everything else inherits them:

1. **Stop the device plugin advertising 32 GPUs.** Whatever the count should be,
   32 is not it, and it is what lets the scheduler oversubscribe the card. This
   is the single highest-leverage change on the list.
2. **Set limits that are not 407%/223% of the node.** Overcommitted limits are
   how a memory spike in one pod evicts an unrelated one.
3. **Reap the Failed pods.** Four dead ReplicaSet children have been sitting for
   9 days. They cost nothing but they hide the next real failure in the noise —
   the same "a broken thing that looks like a normal thing" pattern the rest of
   this codebase's audits keep finding.

None of that needs to happen before the companion service exists. The router
design assumes the node is unreliable, which is what the evidence says it is.
