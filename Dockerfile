FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed AS opencda

ARG USER=opencda
ARG UID=1000 # default uid
ARG HOME=/home/${USER}
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN userdel -r ubuntu && useradd -m -u ${UID} -s /bin/bash ${USER} -d ${HOME}
ENV XDG_RUNTIME_DIR=/tmp/runtime-${USER}
RUN mkdir -p $XDG_RUNTIME_DIR && \
    chmod 700 $XDG_RUNTIME_DIR && \
    ln -sf /usr/bin/python3 /usr/bin/python

ARG PROTOC_VERSION=33.0
ARG PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \

        libsm6=2:1.2.3-1build3 \
        libxext6=2:1.3.4-1build2 \
        libxrender1=1:0.9.10-1.1build1 \
        libvulkan1=1.3.275.0-1build1 \
        libgl1=1.7.0-1build1 \
        mesa-vulkan-drivers=25.0.7-0ubuntu0.24.04.2 \

        curl=8.5.0-2ubuntu10.6 \
        unzip=6.0-28ubuntu4.1 \

        libjpeg-dev=8c-2ubuntu11 \
        libtiff6=4.5.1+git230720-4ubuntu2.4 \

        python3-pip=24.0+dfsg-1ubuntu1.3 \
        python3-dev=3.12.3-0ubuntu2.1 \

        vulkan-tools=1.3.275.0+dfsg1-1 \
 
        libglib2.0-0=2.80.0-6ubuntu3.4 \
        nano \
        ffmpeg \
    && \

    curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP} && \
    unzip -o ${PROTOC_ZIP} -d /usr/local && \
    rm -f ${PROTOC_ZIP} && \

    apt-get purge -y curl unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

USER ${USER}
WORKDIR ${HOME}/cavise/opencda

# Python Version: 3.12.3
COPY opencda/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    python3 -m pip install --no-cache-dir --break-system-packages spconv-cu126==2.3.8
