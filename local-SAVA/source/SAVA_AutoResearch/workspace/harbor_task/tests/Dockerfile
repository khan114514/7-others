FROM python:3.10-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN useradd -m -u 1001 candidate && mkdir -p /workspace/solution
COPY . /tests
RUN chmod 700 /tests/private_cases /tests/private_targets && chmod 600 /tests/private_cases/* /tests/private_targets/* \
    && chmod 755 /tests/test.sh /tests/candidate_worker.py
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["sleep", "infinity"]
