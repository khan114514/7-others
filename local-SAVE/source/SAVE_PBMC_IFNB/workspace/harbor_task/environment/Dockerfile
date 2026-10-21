FROM python:3.10.21-slim-bookworm@sha256:2559be987fd64d61badbdafd303ea58a9ccab36d6c3c08bce219e762177d2eca
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap libgomp1 && rm -rf /var/lib/apt/lists/*
COPY environment/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/runtime && /opt/runtime/bin/pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
ENV PATH=/opt/runtime/bin:$PATH PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba XDG_CACHE_HOME=/tmp/cache
WORKDIR /workspace
COPY environment/starter /workspace/starter
COPY environment/public_assets /workspace/public_assets
COPY environment/public_eval /tests
COPY solution /workspace/solution
RUN useradd -m -u 1000 candidate && chown -R root:root /workspace/starter /workspace/public_assets /tests && chmod -R a-w /workspace/starter /workspace/public_assets /tests && chown -R candidate:candidate /workspace/solution
USER candidate
