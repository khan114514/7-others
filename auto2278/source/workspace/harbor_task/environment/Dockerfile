FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
RUN pip install --no-cache-dir torch==2.5.1 numpy==2.2.6 \
    && useradd -m -u 1001 agent && useradd -m -u 1002 kernel
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
COPY starter /workspace/environment/starter
COPY public_eval /workspace/environment/public_eval
COPY public_assets /workspace/environment/public_assets
COPY instruction.md task.toml /workspace/
RUN mkdir -p /workspace/solution /workspace/public-results \
    && cp /workspace/environment/starter/method.py /workspace/solution/method.py \
    && cp /workspace/environment/starter/solve.sh /workspace/solution/solve.sh \
    && chown -R root:root /workspace/environment /workspace/instruction.md /workspace/task.toml \
    && chmod -R a-w /workspace/environment /workspace/instruction.md /workspace/task.toml \
    && chown -R agent:agent /workspace/solution /workspace/public-results \
    && chown root:root /workspace/solution/solve.sh && chmod 555 /workspace/solution/solve.sh
USER agent
