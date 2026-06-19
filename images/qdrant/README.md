# qdrant (16K-page build)

Qdrant built from source so it runs on **16K-page** kernels. The official
`qdrant/qdrant` image bundles jemalloc for 4K pages and aborts with
`<jemalloc>: Unsupported system page size` on both our targets:

- **Pi 5 / Rocky 10** — boots the `+16k` BCM2712 kernel (`getconf PAGE_SIZE` = 16384).
- **Mac mini M2 / Asahi** — Apple Silicon is natively 16K.

The fix is a single build-time env: `JEMALLOC_SYS_WITH_LG_PAGE=16`, which builds
jemalloc for pages up to 64K — one binary works on 4K / 16K / 64K.

**Memory:** Qdrant's release profile uses fat LTO, whose final link needs >8 GB
and OOM-kills on an 8 GB host (the Pi, or an 8 GB build VM). The Dockerfile
overrides it to thin LTO + parallel codegen units. If an 8 GB host still OOMs,
set `CARGO_PROFILE_RELEASE_LTO=off`.

## Build

Ad-hoc on each host (no registry push — slow on the Pi, faster on the M2):

```bash
# default (CARGO_BUILD_JOBS=2, Pi-safe)
docker build -t qdrant:16k images/qdrant

# faster on a roomier box
docker build -t qdrant:16k \
  --build-arg CARGO_BUILD_JOBS=6 \
  --build-arg QDRANT_VERSION=v1.18.2 \
  images/qdrant
```

The deploy playbook (containers-at-home) builds this in place of pulling the
upstream image. `app_versions['docs-bridge-qdrant'].image` must point at the
locally built tag (e.g. `qdrant:16k`).

## Verify it runs (the whole point)

```bash
docker run --rm -p 6333:6333 qdrant:16k &
curl -fsS localhost:6333/readyz && echo OK   # must NOT abort with a page-size error
```
