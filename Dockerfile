# Immutable base; Python packages are pinned in env/*.lock.txt.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential cmake ninja-build git \
    curl ca-certificates libaio-dev libnuma-dev pkg-config zlib1g-dev \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/aisbench-ucm-repro
COPY . .
RUN python3 scripts/setup.py sources \
    && python3 scripts/setup.py client \
    && python3 scripts/setup.py server --jobs 4
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MODEL_PATH=/models/model
ENTRYPOINT ["python3", "scripts/run.py"]
CMD ["plan"]
