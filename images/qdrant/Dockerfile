# Qdrant built from source with jemalloc configured for LARGE PAGES.
#
# Why: the official qdrant/qdrant image bundles jemalloc compiled for 4K pages
# and aborts ("Unsupported system page size") on 16K-page kernels. BOTH our target
# hosts use 16K pages — the Pi 5 boots the +16k BCM2712 kernel, and Apple Silicon
# under Asahi is natively 16K — so the stock image cannot run on either.
#
# The fix: JEMALLOC_SYS_WITH_LG_PAGE=16 builds jemalloc for pages up to 64K, so a
# single binary runs on 4K / 16K / 64K. Build is ad-hoc on each host (no registry
# push); see images/qdrant/README.md.
ARG QDRANT_VERSION=v1.18.2
ARG CARGO_BUILD_JOBS=2          # Pi-safe default; override on roomier hosts

FROM rust:1-bookworm AS builder
ARG QDRANT_VERSION
ARG CARGO_BUILD_JOBS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang libclang-dev cmake \
        protobuf-compiler libprotobuf-dev pkg-config \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${QDRANT_VERSION}" \
        https://github.com/qdrant/qdrant.git .
# The one line that matters: jemalloc for <=64K pages (covers the Pi/M2 16K).
# Qdrant's release profile uses fat LTO + codegen-units=1, whose final-binary
# link needs >8GB and OOM-kills on an 8GB box (both the Pi and an 8GB build VM).
# Override to thin LTO (keeps most perf, far lower peak RAM) and parallel codegen.
# If an 8GB host still OOMs, drop to CARGO_PROFILE_RELEASE_LTO=off.
ENV JEMALLOC_SYS_WITH_LG_PAGE=16 \
    CARGO_PROFILE_RELEASE_LTO=thin \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
RUN cargo build --release --bin qdrant --jobs "${CARGO_BUILD_JOBS}"

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /qdrant
COPY --from=builder /src/target/release/qdrant /qdrant/qdrant
COPY --from=builder /src/config /qdrant/config
# storage (./storage) and snapshots (./snapshots) are bind-mounted at runtime;
# their defaults live in config/config.yaml. Matches the official image layout.
EXPOSE 6333 6334
ENTRYPOINT ["/qdrant/qdrant"]
