FROM python:3.10-slim-bookworm
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends nodejs npm bubblewrap git ca-certificates procps sudo \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex@0.160.0 && npm cache clean --force
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN useradd -m -u 1000 actor && useradd -m -u 1001 evaluator-candidate && mkdir -p /workspace/solution /workspace/.git /workspace/.agents /workspace/.codex /logs/agent /logs/artifacts /opt/public_eval/results
COPY starter /workspace/environment/starter
COPY starter /workspace/solution
COPY public_assets /workspace/environment/public_assets
COPY public_eval /opt/public_eval
COPY instruction.md task.toml /workspace/
RUN chmod -R a-w /workspace/environment /opt/public_eval /workspace/instruction.md /workspace/task.toml \
    && chmod 700 /opt/public_eval/targets && chmod 600 /opt/public_eval/targets/* \
    && chmod 755 /opt/public_eval/evaluate /opt/public_eval/results \
    && chown actor:actor /workspace/solution /workspace/solution/method.py /home/actor \
    && chmod 444 /workspace/solution/solve.sh \
    && printf 'actor ALL=(root) NOPASSWD: /opt/public_eval/evaluate\n' > /etc/sudoers.d/valuation \
    && chmod 440 /etc/sudoers.d/valuation
WORKDIR /workspace
USER root
ENTRYPOINT ["python", "-I", "/opt/public_eval/entrypoint.py"]
CMD ["sleep", "infinity"]
