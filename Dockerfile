FROM docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd AS opencda

ARG USER=opencda
ARG UID=1000 # default uid
ARG HOME=/home/${USER}
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN userdel -r ubuntu && useradd -l -m -u ${UID} -s /bin/bash ${USER} -d ${HOME}
ENV XDG_RUNTIME_DIR=/tmp/runtime-${USER}
RUN mkdir -p $XDG_RUNTIME_DIR && \
    chmod 700 $XDG_RUNTIME_DIR && \
    ln -sf /usr/bin/python3 /usr/bin/python

ARG PROTOC_VERSION=34.1
ARG PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6=2:1.2.* \
        libxext6=2:1.3.* \
        libxrender1=1:0.9.* \
        libvulkan1=1.3.* \
        libgl1=1.7.* \
        mesa-vulkan-drivers=25.2.* \
        curl=8.5.* \
        unzip=6.0-* \
        libjpeg-dev=8c-* \
        libtiff6=4.5.* \
        python3-pip=24.0+* \
        python3-dev=3.12.* \
        vulkan-tools=1.3.* \
        libglib2.0-0t64=2.80.* \
    && \
    curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP} && \
    unzip -o ${PROTOC_ZIP} -d /usr/local && \
    rm -f ${PROTOC_ZIP} && \
    apt-get purge -y curl unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

USER ${USER}
ENV PATH="${HOME}/.local/bin:${PATH}"
WORKDIR ${HOME}/cavise/opencda

# Python Version: 3.12.3
COPY opencda/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 && \
    python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt
