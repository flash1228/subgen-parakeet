# ---- Stage 1: builder -------------------------------------------------------
# Use the -devel- flavor so nvcc (CUDA toolkit compiler) is on PATH for the
# libtranscribe CUDA build. The -base- flavor is runtime-only and causes
# `cmake -DTRANSCRIBE_CUDA=ON` to fail with "no CMAKE_CUDA_COMPILER".
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# libtranscribe is provided by the transcribe-cpp[cu12] wheel.
# No source compilation is performed to keep the image lean.

# ---- Stage 2: runtime -------------------------------------------------------
# The -base- flavor is much smaller (~200 MB) and carries the CUDA driver
# compatibility libs / cuBLAS / cuSPARSE / cuFFT that libtranscribe links
# against at runtime — everything needed to actually run inference.
FROM nvidia/cuda:12.6.1-base-ubuntu22.04

# Prevent apt post-install prompts (tzdata etc.) from stalling the build.
ENV DEBIAN_FRONTEND=noninteractive

# Copy the CUDA-backed libtranscribe shared library built in stage 1.
# libtranscribe is provided by the transcribe-cpp[cu12] wheel; no copy needed

RUN ldconfig && \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip curl gosu tzdata \
        libsndfile1 libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
        python-is-python3 \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Torch — optional, only used for the CUDA cache clear path. Keep on a separate
# layer so it can be removed without rebuilding the app deps.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# App dependencies — only rebuilds when requirements.txt changes.
# transcribe-cpp[cu12] pulls the CUDA 12 native provider wheel.
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U --no-cache-dir "transcribe-cpp[cu12]" "transcribe-cpp-native-cu12" -r /requirements.txt && \
    apt-get purge -y --auto-remove python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /subgen

# App files last — changes here don't bust the layers above.
COPY launcher.py subgen.py language_code.py parakeet_gguf.py /subgen/

RUN mkdir -p /cache /subgen/models && chmod 777 /cache /subgen/models

ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
