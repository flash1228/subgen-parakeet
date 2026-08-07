FROM nvidia/cuda:12.6.1-base-ubuntu22.04

# Apt packages — own layer so pip changes don't re-run apt.
# Adds build tools (cmake, g++, git, libopenblas-dev) for building libtranscribe
# from source, since transcribe-cpp-native wheels aren't reliably published yet.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip curl gosu tzdata \
        libsndfile1 libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
        python-is-python3 \
        git cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Build libtranscribe from source against this image's CUDA 12.6 toolkit.
# Pinned to a known-good ref of handy-computer/transcribe.cpp; bump ARG to
# track upstream. The shared library is installed to /usr/local/lib so it's
# found via TRANSCRIBE_LIBRARY at runtime.
ARG TRANSCRIBE_CPP_REF=main
RUN --mount=type=cache,target=/root/.cache/apt,sharing=locked \
    git clone --depth 1 --branch ${TRANSCRIBE_CPP_REF} \
        https://github.com/handy-computer/transcribe.cpp.git /tmp/transcribe.cpp && \
    cmake -S /tmp/transcribe.cpp -B /tmp/transcribe.cpp/build \
        -DTRANSCRIBE_BUILD_SHARED=ON \
        -DTRANSCRIBE_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/transcribe.cpp/build --target transcribe -- -j"$(nproc)" && \
    cmake --install /tmp/transcribe.cpp/build --prefix /usr/local && \
    ldconfig && \
    rm -rf /tmp/transcribe.cpp

# Torch — optional, only used for the CUDA cache clear path. Keep on a separate
# layer so it can be removed without rebuilding the app deps.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# App dependencies — only rebuilds when requirements.txt changes
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U --no-cache-dir --no-deps -r /requirements.txt 2>&1 | tee /tmp/pip.log && \
    python3 -m pip install -U --no-cache-dir "transcribe-cpp[cu12]" -r /requirements.txt && \
    apt-get purge -y --auto-remove python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /subgen

# App files last — changes here don't bust the layers above
COPY launcher.py subgen.py language_code.py parakeet_gguf.py /subgen/

RUN mkdir -p /cache /subgen/models && chmod 777 /cache /subgen/models

ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1 \
    # Tell transcribe-cpp where to find the shared library we just built.
    TRANSCRIBE_LIBRARY=/usr/local/lib/libtranscribe.so

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
