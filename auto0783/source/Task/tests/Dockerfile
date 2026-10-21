FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN groupadd -g 1000 verifier && useradd -u 1000 -g 1000 -m verifier && mkdir -p /workspace/solution /run/bifrost/queue /logs/verifier && chown -R 1000:1000 /workspace /run/bifrost/queue /logs/verifier
COPY queue_client.py /opt/bifrost/queue_client.py
COPY grade.py test.sh /tests/
COPY hidden-chunks/ /tests/hidden-chunks/
COPY prepare_hidden.py /tests/prepare_hidden.py
RUN python -I /tests/prepare_hidden.py
RUN chmod a+rx /tests/test.sh && chmod -R a+rX /tests/hidden
ENV HOME=/home/verifier PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH=/opt/bifrost
USER 1000:1000
WORKDIR /workspace/solution
CMD ["sleep","infinity"]
