# Benchmark 8: BERTopic topic inference (08_infer_bertopic.py)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    NUMBA_CACHE_DIR=/opt/numba_cache

WORKDIR /app

# CPU-only torch (the default PyPI wheel bundles several GB of CUDA libraries)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-compile UMAP/HDBSCAN's numba functions (including UMAP's transform, which
# inference uses) into a cache so timings don't include one-off JIT compilation.
# Writable by any user (--user runs).
RUN mkdir -p $NUMBA_CACHE_DIR && python -c "\
import numpy as np; from umap import UMAP; from hdbscan import HDBSCAN, approximate_predict; \
u = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine').fit(np.random.rand(2000, 384)); \
x = u.transform(np.random.rand(500, 384)); \
h = HDBSCAN(min_cluster_size=10, prediction_data=True).fit(u.embedding_); approximate_predict(h, x)" \
    && chmod -R 777 $NUMBA_CACHE_DIR

# The embedding model (all-MiniLM-L6-v2) is stored inside bertopic_model.pkl
COPY 08_infer_bertopic.py bertopic_model.pkl Comments.xml ./

# extra arguments to `docker run` are passed to the script, e.g. -n 5000
ENTRYPOINT ["python", "08_infer_bertopic.py", "Comments.xml"]
