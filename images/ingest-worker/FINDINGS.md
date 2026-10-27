# images/ingest-worker — findings register

What was found building the worker image, and how each item was resolved. Numbering
is stable. The operating manual is [README.md](README.md).

## Findings & peculiarities (and how we resolved them)

### 1. BuildKit refuses a blank `FROM ${ARG}` even in a stage the target skips
With `ARG MODELS_IMAGE` (no default) and a `FROM ${MODELS_IMAGE} AS models` stage that
only `runtime-cpu` uses, building `--target runtime-service` failed on 2026-10-27
(Docker 29.8): "base name (${MODELS_IMAGE}) should not be blank". BuildKit validates
every `FROM` before it prunes the stages the target does not need.
**Resolution:** `MODELS_IMAGE` defaults to a placeholder that cannot exist,
`localhost/runtime-cpu-needs-build-arg-models-image:unset`. `runtime-service` never
resolves it; `runtime-cpu` without the arg fails pulling it, and the name says why.
It is not a default model: no build can produce an image from it.
