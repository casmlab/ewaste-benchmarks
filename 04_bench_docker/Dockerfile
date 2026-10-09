# Benchmark 4: logistic regression inference (04_ams-label-comparisons.py)
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

COPY 04_ams-label-comparisons.py congress115.csv.gz lr_0.9.pkl ./

CMD ["python", "04_ams-label-comparisons.py"]
