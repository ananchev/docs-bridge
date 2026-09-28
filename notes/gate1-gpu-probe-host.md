# Gate 1 §2.3 — the GPU probe host

> **Run 2026-09-28 on a `g4dn.xlarge` (T4): fp16 0.54 s/query, 0.29 s bucketed at 8,
> against 14.2 s on CPU.** Results and decision: `gate1-runbook.md` §2.3; plain
> English: `gate1-findings-explained.md` test 5. The harness's FAIL verdict is an
> artifact of the INT8 reference (fp32 fails it identically) — see there.

What to launch, what to run on it, and what to bring back. Scoped to **one
afternoon on one instance, then terminate it.** This is the cheapest possible test
of the only question that matters at this stage: *is a GPU fast enough to make the
11 s query — and the whole queueing argument in `multi-user-rollout-plan.md` §0.1 —
go away?* If the answer disappoints, the entire GPU track dies here for the price
of a few dollars of instance time and zero repo changes.

## What the probe does NOT need

The cross-encoder is a pure function of (query, passage), so this host needs
**no corpus, no Qdrant, no manifest, no docs-bridge image, no config.yaml, no
ingest, no EBS data volume, and no inbound port but SSH.** Two files are scp'd in:

| File | What it is | Size |
|---|---|---|
| `fixture.json` | Gate 1 §2.2 output: 8 real queries, their real RRF candidate passages, and the CPU-INT8 baseline scores. **Not in git** — on vhost2 at `~/gate1/out/fixture.json` | 1.0 MB |
| `tools/eval/rerank_ab.py` | the probe — standalone, imports only `torch` + `transformers` | 18 KB |

That is the whole reason §2.2 exists before §2.3.

## Instance

**x86_64, not Graviton.** The probe itself is torch and arch-agnostic, but the
decision it feeds is not: `onnxruntime-gpu` has **no aarch64 wheels below 1.29.0**
(plan §6.3), so a g5g target would force ORT nine minor versions forward on its
own. Measure on the architecture the GPU path would actually ship on.

| Priority | Instance | GPU | vCPU / RAM | Why |
|---|---|---|---|---|
| **Primary** | `g4dn.xlarge` | T4, 16 GB | 4 / 16 GiB | Cheapest NVIDIA on EC2 and the plan's headline projection (~0.5 s/query). If the T4 clears the bar, nothing more expensive needs justifying. |
| **Second point** | `g6.xlarge` | L4, 24 GB | 4 / 16 GiB | The plan's other projection (~0.3 s). Ada, so it also supports bf16. Substitute `g5.xlarge` (A10G, 24 GB) if g6 is not in your region. |

One instance is enough to decide; two give the price/performance slope. Both are
4 vCPU, which is deliberate — we are not measuring CPU here, and the fixture
already carries the CPU baseline from vhost2.

**Quota:** this consumes the *Running On-Demand All G and VT instances* vCPU quota
(4 vCPU for either instance). That is the quota already granted.

## AMI

Use a **Deep Learning AMI with the driver and PyTorch already installed** — it
removes the two things that actually waste an afternoon (NVIDIA driver, CUDA torch
wheel):

- Preferred: **`Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)`**
  — driver + CUDA + torch preinstalled; only `transformers` is missing.
- Acceptable: **`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)`**
  — driver + container toolkit, no torch, so add `pip install torch --index-url
  https://download.pytorch.org/whl/cu124` (~2.5 GB).

**As actually provisioned (2026-09-28):** the Ubuntu 22.04 variant was not
available, so the host runs **`Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12
(Ubuntu 24.04)`** — torch 2.12.1+cu130 on Python 3.13, in the `/opt/pytorch` venv
(not on the system `python3`). The exact commands for this host are in
`gate1-runbook.md` step 5, which supersedes the setup below.

Resolve the current image id at launch rather than pinning one from this file:

```bash
aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*' \
  --query 'reverse(sort_by(Images,&CreationDate))[:5].[ImageId,Name]' --output table
```

**Storage:** the PyTorch DLAMI wants a large root volume (~110 GiB gp3) — take the
AMI default and do not shrink it. Model weights add ~2.3 GB (bge-reranker-v2-m3
fp32). No second volume.

**Network:** SSH (22) inbound from your IP only; default egress (needed to pull the
weights from HuggingFace — the model is public, no token). Nothing else. No load
balancer, no target group; this host serves no traffic.

## Setup

```bash
# on the PyTorch DLAMI the torch env is already active; otherwise activate it
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -q transformers            # the only missing dep
nvidia-smi                             # confirm the driver sees the GPU
```

Then from your workstation (the fixture comes from vhost2, not the repo):

```bash
scp -3 vhost2:gate1/out/fixture.json <gpu-host>:
scp tools/eval/rerank_ab.py <gpu-host>:
```

## The three runs

fp32 first, on purpose. It separates the two things a single fp16 run would
confound — *what the GPU costs in numerics* from *what INT8 already costs* — and on
a GPU it is nearly free:

```bash
# 1. fp32 on CUDA: the precision reference. Any ranking difference against the
#    fixture's baseline column here is INT8's doing, not the GPU's.
python rerank_ab.py --fixture fixture.json --backend torch \
    --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision fp32 \
    --repeats 10 --json gpu-fp32.json | tee gpu-fp32.txt

# 2. fp16 on CUDA: the actual candidate — the number the decision rests on.
python rerank_ab.py --fixture fixture.json --backend torch \
    --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision fp16 \
    --repeats 10 --json gpu-fp16.json | tee gpu-fp16.txt

# 3. bf16 — ONLY on L4/A10G (Ada/Ampere). The T4 is Turing and has no bf16, so
#    skip this run there; it would fall back and mislead.
python rerank_ab.py --fixture fixture.json --backend torch \
    --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision bf16 \
    --repeats 10 --json gpu-bf16.json | tee gpu-bf16.txt
```

`--repeats 10` because a GPU run is short; the harness warms up before timing, so
CUDA kernel autotune and the lazy context init are not counted.

Bring back the three `.json` files and the three `.txt` logs. **Then terminate the
instance** — nothing on it is needed again.

## What decides it

The harness prints the verdict itself, against the plan §2.4 bar: **top-3
ordered-identical, top-6 Jaccard ≥ 0.83** versus the CPU-INT8 baseline, with every
disagreement adjudicated by the deeper-pool oracle rather than assumed to be a
regression — fp16 is strictly *more* precise than INT8, so a difference may well be
an improvement, and the oracle is what tells the two apart.

Read the outcome as:

| fp16 latency at `top_n=40` | Reading |
|---|---|
| **≲ 1 s** | The GPU path wins outright. Sub-second queries mean §0.1's queueing arithmetic stops being a design input at all; proceed to §6.3 packaging (the ~1 day CUDA base image is the real work) and x86_64. |
| **1–4 s** | Compare against §3's smaller-reranker number on the *existing* CPU box. If `bge-reranker-base` lands in the same range, take it: no new hardware, no CUDA packaging, no architecture change. |
| **≳ 4 s, or the ranking bar fails** | The GPU track dies. Size a CPU pool on C from §2.1 and revisit only if the corpus or `top_n` grows. |

Cost is a few dollars either way (roughly $0.5–1.0/instance-hour in us-east-1 —
verify at launch), which is the entire point of testing physics before packaging.
