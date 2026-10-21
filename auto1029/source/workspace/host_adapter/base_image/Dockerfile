FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap ca-certificates && rm -rf /var/lib/apt/lists/*
RUN groupadd -g 1000 candidate && useradd -u 1000 -g 1000 -m candidate && mkdir -p /workspace/solution && chown -R 1000:1000 /workspace
COPY codex codex-code-mode-host rg /usr/local/bin/
ENV HOME=/home/candidate PATH=/usr/local/bin:/usr/bin:/bin
USER 1000:1000
WORKDIR /workspace
CMD ["sleep","infinity"]
