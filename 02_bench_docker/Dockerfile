# Benchmark 2: Linear regressions of engagement on post topic (02_random_effects.R)
# rocker/r-ver installs CRAN packages as they were on the R version's release
# date, from Posit Package Manager's prebuilt binaries
FROM rocker/r-ver:4.6.1

RUN install2.r --error --skipinstalled \
    broom dplyr tidyr ggplot2 scales data.table future furrr parallelly purrr tictoc \
    && rm -rf /tmp/downloaded_packages

WORKDIR /app
# The real data file. The [v] makes the pattern optional, so the image can be
# built without it, using synthetic data instead (below).
COPY 02_random_effects.R generate_synthetic_data.R all_posts_w_topic_and_candidate.cs[v] ./

# --build-arg SYNTHETIC=1 generates synthetic data in place of the real file
# (see README). Otherwise the build fails if the real file is missing.
ARG SYNTHETIC=0
RUN if [ "$SYNTHETIC" = 1 ]; then \
        Rscript generate_synthetic_data.R synthetic_all_posts_w_topic_and_candidate.csv; \
    elif [ ! -f all_posts_w_topic_and_candidate.csv ]; then \
        echo "all_posts_w_topic_and_candidate.csv is missing: copy it into this folder, or build with --build-arg SYNTHETIC=1" >&2; \
        exit 1; \
    fi

# writable by any user, so the container can run with --user
RUN mkdir -p /app/output && chmod 777 /app/output

# Arguments to `docker run` after the image name are passed to the script:
# [input file] [max rows]
ENTRYPOINT ["Rscript", "02_random_effects.R"]
