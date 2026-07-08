# Pin to amd64: pre-built libarrow binaries are available for linux/amd64.
# On Apple Silicon, Docker Desktop runs this via Rosetta transparently.
FROM --platform=linux/amd64 rocker/r-ver:4.4.2

# System libraries required by the arrow R package
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# BiocManager -> limma; arrow for reading Parquet files
RUN Rscript -e "install.packages('BiocManager', repos='https://cloud.r-project.org')" \
 && Rscript -e "BiocManager::install('limma', ask=FALSE, update=FALSE)" \
 && Rscript -e "install.packages('arrow', repos='https://cloud.r-project.org')"
