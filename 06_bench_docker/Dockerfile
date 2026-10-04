# Benchmark 6: LDA topic inference with MALLET (06_ams-unigram-lda-inference-MALLET.py)
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
    && sed -i 's/^MEMORY=1g$/MEMORY=${MALLET_MEMORY:-1g}/' /opt/mallet-2.0.8/bin/mallet

ENV MALLET_DIR=/opt/mallet-2.0.8/bin/mallet

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Hemphill et al.'s published inferencer (models/best/lda/ in their repo). Their
# congress115_train.mallet, whose pipe --use-pipe-from needs, isn't published,
# so rebuild an equivalent one: a single document listing the inferencer's
# vocabulary in word-ID order, imported with MALLET's default import settings.
COPY lda_tw_50_unigram_congress115_inferencer.mallet export_vocabulary.bsh model/
RUN java -cp "/opt/mallet-2.0.8/class:/opt/mallet-2.0.8/lib/*" bsh.Interpreter \
        model/export_vocabulary.bsh model/lda_tw_50_unigram_congress115_inferencer.mallet /tmp/vocabulary.txt \
    && $MALLET_DIR import-file --input /tmp/vocabulary.txt \
        --line-regex '^([^,]*),([^,]*),(.*)$' --name 1 --label 0 --data 3 \
        --keep-sequence --output model/congress115_vocabulary_pipe.mallet \
    && rm /tmp/vocabulary.txt

COPY 06_ams-unigram-lda-inference-MALLET.py congress115.csv.gz russell_processed_0.9.csv.gz ./

# writable by any user, so the container can run with --user
RUN mkdir -p /app/output && chmod 777 /app/output

CMD ["python", "06_ams-unigram-lda-inference-MALLET.py"]
