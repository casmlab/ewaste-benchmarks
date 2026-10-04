# Benchmark 3: logistic regression training (03_ams-create-best-classifiers.py)
# Pinned to Python 3.6 for compatibility with scikit-learn 0.20.x / pandas 0.24.x
FROM python:3.6-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade "pip<22" && pip install -r requirements.txt

COPY 03_ams-create-best-classifiers.py russell_processed_0.9.csv.gz ./

CMD ["python", "03_ams-create-best-classifiers.py"]
