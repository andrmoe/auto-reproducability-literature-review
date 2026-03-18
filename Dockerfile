FROM ubuntu:24.04
USER root
WORKDIR /root
RUN apt-get update && apt install -y --no-install-recommends curl python3 python3-pip
#RUN curl -fsSL https://claude.ai/install.sh | bash
