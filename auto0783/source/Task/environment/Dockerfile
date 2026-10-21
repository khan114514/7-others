FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN groupadd -g 1000 candidate && useradd -u 1000 -g 1000 -m candidate && mkdir -p /workspace/solution /run/bifrost/queue /opt/bifrost/starter && chown -R 1000:1000 /workspace /run/bifrost/queue
COPY tools/ /opt/bifrost/public-cli/
RUN python -I /opt/bifrost/public-cli/install_cli.py
COPY runtime/ /opt/bifrost/
COPY public_eval/ /opt/bifrost/public_eval/
COPY starter/ /opt/bifrost/starter/
RUN cd /opt/bifrost/starter && python method.py && chmod -R a+rX /opt/bifrost && chmod a+rx /usr/local/bin/codex /usr/local/bin/codex-code-mode-host /usr/local/bin/rg
ENV HOME=/home/candidate PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH=/opt/bifrost BIFROST_LANE=gpt
USER 1000:1000
WORKDIR /workspace/solution
ENTRYPOINT ["python","/opt/bifrost/entrypoint.py"]
CMD ["sleep","infinity"]
