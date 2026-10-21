FROM ubuntu:22.04
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY runtime_client.py /tests/runtime_client.py
COPY test.sh /tests/test.sh
RUN chmod 700 /tests/test.sh
CMD ["/bin/bash"]
