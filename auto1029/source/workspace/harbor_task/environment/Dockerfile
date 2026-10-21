FROM heima-agent-base:20261005
USER root
COPY entrypoint.py local_model_proxy.py queue_client.py evaluate.py run_codex.py snapshot_files.py /opt/latentcli/
COPY starter.py starter_notes.json /opt/latentcli/
COPY instruction.md /workspace/instruction.md
COPY public_assets/ /workspace/public_assets/
RUN mkdir -p /workspace/solution /run/latentcli/queue /run/heima-verifier/queue /tests /solution /logs/agent /logs/verifier /logs/artifacts && chown -R 1000:1000 /workspace/solution /run/latentcli /logs/agent && chmod 0700 /tests /solution /logs/verifier /run/heima-verifier/queue
USER 1000:1000
WORKDIR /workspace/solution
ENTRYPOINT ["python","/opt/latentcli/entrypoint.py"]
CMD ["sleep","infinity"]
