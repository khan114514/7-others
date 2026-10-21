FROM heima-agent-base:20261005
USER root
COPY entrypoint.py local_model_proxy.py queue_client.py grade.py snapshot_files.py /opt/latentcli/
COPY test.sh /tests/test.sh
RUN chmod 0700 /tests && chmod 0755 /tests/test.sh && mkdir -p /workspace/solution /run/latentcli/queue /run/heima-verifier/queue /logs/agent /logs/verifier /logs/artifacts && chown -R 1000:1000 /workspace/solution /run/latentcli /logs/agent && chmod 0700 /run/heima-verifier/queue /logs/verifier
WORKDIR /workspace
ENTRYPOINT ["python","/opt/latentcli/entrypoint.py"]
CMD ["sleep","infinity"]
