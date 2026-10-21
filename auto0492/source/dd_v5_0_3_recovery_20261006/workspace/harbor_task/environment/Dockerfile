FROM node:22.16.0-bookworm-slim AS node_runtime
FROM ubuntu:22.04
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DD_RUNTIME_SOCKET=/run/dd/public.sock
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl git ripgrep && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN node /usr/local/lib/node_modules/npm/bin/npm-cli.js install -g @openai/codex@0.160.0 && codex --version
WORKDIR /workspace
COPY instruction.md /workspace/instruction.md
COPY starter /workspace/environment/starter
COPY public_eval /workspace/environment/public_eval
RUN mkdir -p /workspace/solution /workspace/agent-output /installed-agent && chmod +x /workspace/environment/public_eval/solve.sh
COPY bootstrap.sh /usr/local/bin/dd-bootstrap
ENTRYPOINT ["/usr/local/bin/dd-bootstrap"]
CMD ["/bin/bash"]
