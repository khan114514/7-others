FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN pip install --no-cache-dir torch==2.5.1 numpy==2.2.6 && useradd -m -u 1002 kernel
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /tests
COPY . /tests/
RUN chmod 600 /tests/protocol.json && chmod 555 /tests/test.sh \
    && mkdir -p /workspace/solution /logs/verifier
