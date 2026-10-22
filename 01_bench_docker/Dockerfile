# Benchmark 1: Synthetic data generation with an LLM (01_synthetic_data.py)
FROM python:3.11-slim-bookworm

ARG TARGETARCH=amd64
ARG OLLAMA_VERSION=v0.40.6
# The LLM baked into the image; runs don't download models
ARG MODEL=gemma3:4b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    OLLAMA_MODELS=/opt/ollama/models \
    HOME=/tmp

# Ollama, CPU-only: the release bundles GPU libraries (CUDA, ROCm, Vulkan, MLX)
# that add over 1 GB and aren't used here
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
       | tar --zstd -x -C /usr/local \
    && find /usr/local/lib/ollama -mindepth 1 -maxdepth 1 -type d \
       \( -iname '*cuda*' -o -iname '*rocm*' -o -iname '*vulkan*' -o -iname '*mlx*' -o -iname '*jetpack*' \) \
       -exec rm -rf {} + \
    && apt-get purge -y curl zstd && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Bake the LLM into the image
RUN mkdir -p $OLLAMA_MODELS \
    && (ollama serve >/dev/null 2>&1 &) \
    && for i in $(seq 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done \
    && ollama pull "$MODEL" \
    && chmod -R a+rX $OLLAMA_MODELS

WORKDIR /app

# CPU-only torch (the default PyPI wheel bundles several GB of CUDA libraries)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so runs don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY 01_synthetic_data.py posts.csv ./

# writable by any user, so the container can run with --user
RUN mkdir -p /app/output && chmod 777 /app/output

# extra arguments to `docker run` are passed to the script, e.g. -n 2
ENTRYPOINT ["python", "01_synthetic_data.py", "posts.csv"]
