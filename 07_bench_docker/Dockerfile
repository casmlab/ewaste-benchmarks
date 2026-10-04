# Benchmark 7: BERTopic model training (07_fit_bertopic.py)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    NUMBA_CACHE_DIR=/opt/numba_cache

WORKDIR /app

# CPU-only torch (the default PyPI wheel bundles several GB of CUDA libraries)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so runs don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

# Pre-compile UMAP/HDBSCAN's numba functions into a cache so the fit timing
# doesn't include one-off JIT compilation. Writable by any user (--user runs).
RUN mkdir -p $NUMBA_CACHE_DIR && python -c "\
import numpy as np; from umap import UMAP; from hdbscan import HDBSCAN; \
x = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine').fit_transform(np.random.rand(2000, 384)); \
HDBSCAN(min_cluster_size=10, prediction_data=True).fit(x)" \
    && chmod -R 777 $NUMBA_CACHE_DIR

COPY 07_fit_bertopic.py Comments.xml ./

# writable by any user, so the container can run with --user
RUN mkdir -p /app/output && chmod 777 /app/output

# extra arguments to `docker run` are passed to the script, e.g. -n 5000
ENTRYPOINT ["python", "07_fit_bertopic.py", "Comments.xml"]
