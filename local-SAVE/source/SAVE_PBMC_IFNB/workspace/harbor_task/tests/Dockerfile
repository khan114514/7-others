FROM python:3.10.21-slim-bookworm@sha256:2559be987fd64d61badbdafd303ea58a9ccab36d6c3c08bce219e762177d2eca
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/runtime && /opt/runtime/bin/pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
ENV PATH=/opt/runtime/bin:$PATH PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba XDG_CACHE_HOME=/tmp/cache
WORKDIR /workspace
COPY starter /workspace/starter
COPY public_assets /workspace/public_assets
COPY *.py *.json test.sh /tests/
COPY hidden_assets /tests/hidden_assets
RUN chmod -R a-w /workspace/starter /workspace/public_assets /tests
