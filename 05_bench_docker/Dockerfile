# Benchmark 5: LDA topic model training with MALLET (05_ams-unigram-lda-model-MALLET.py)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# MALLET 2.0.8 release (includes compiled classes and lib/ dependencies)
ARG MALLET_SHA256=5b2d6fb9bcf600b1836b09881821a6781dd45a7d3032e61d7500d027a5b34faf
RUN curl -sSL -o /tmp/mallet.tar.gz https://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz \
    && echo "${MALLET_SHA256}  /tmp/mallet.tar.gz" | sha256sum -c - \
    && tar xzf /tmp/mallet.tar.gz -C /opt \
    && rm /tmp/mallet.tar.gz \
    && find /opt/mallet-2.0.8 -name '._*' -delete \
    # the launcher hard-codes 1g of Java heap; let MALLET_MEMORY override it
    && sed -i 's/^MEMORY=1g$/MEMORY=${MALLET_MEMORY:-4g}/' /opt/mallet-2.0.8/bin/mallet

ENV MALLET_DIR=/opt/mallet-2.0.8/bin/mallet

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY 05_ams-unigram-lda-model-MALLET.py congress115_preprocessed.csv.gz ./

# writable by any user, so the container can run with --user
RUN mkdir -p /app/output && chmod 777 /app/output

CMD ["python", "05_ams-unigram-lda-model-MALLET.py"]
