ARG CUDA_VERSION=13.0.3
ARG CUDA_DEVEL_DIGEST=sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
ARG CUDA_RUNTIME_DIGEST=sha256:14f6d08d1cd4a96effbfe3101d0b56326f552c199d05e4979ee0bd616df5811b
ARG UBUNTU_VERSION=24.04
ARG UBUNTU_DIGEST=sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}@${CUDA_RUNTIME_DIGEST} AS runtime-base

ARG USER=opencda
ARG UID=1000
ARG HOME=/home/${USER}

ENV HOME=${HOME}
ENV PATH="/opt/venv/bin:${PATH}"
ENV XDG_RUNTIME_DIR=/tmp/runtime-${USER}
ENV OPENCDA_WORKSPACE=${HOME}/cavise/opencda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.* \
        libsm6=2:1.2.* \
        libxext6=2:1.3.* \
        libxrender1=1:0.9.* \
        libvulkan1=1.3.* \
        libgl1=1.7.* \
        mesa-vulkan-drivers=25.2.* \
        libjpeg-turbo8=2.1.* \
        libtiff6=4.5.* \
        python3=3.12.* \
        vulkan-tools=1.3.* \
        libglib2.0-0t64=2.80.* \
    && \
    rm -rf /var/lib/apt/lists/*

RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi && \
    useradd --create-home --user-group --uid ${UID} --shell /bin/bash ${USER} && \
    mkdir -p ${XDG_RUNTIME_DIR} ${OPENCDA_WORKSPACE} && \
    chmod 700 ${XDG_RUNTIME_DIR} && \
    chown -R ${UID}:${UID} ${XDG_RUNTIME_DIR} ${HOME}


FROM runtime-base AS python-dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-venv=3.12.* && \
    rm -rf /var/lib/apt/lists/*

COPY opencda/requirements.txt opencda/requirements-cuda.txt /tmp/opencda-requirements/

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --no-cache-dir --upgrade \
        pip==26.1.2 \
        setuptools==82.0.1 \
        wheel==0.47.0 \
    && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        -r /tmp/opencda-requirements/requirements.txt


FROM ubuntu:${UBUNTU_VERSION}@${UBUNTU_DIGEST} AS protobuf-builder

ARG PROTOC_VERSION=34.1

ENV PATH="/opt/protobuf-build-venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.* \
        curl=8.5.* \
        python3=3.12.* \
        python3-venv=3.12.* \
        unzip=6.0-* \
    && \
    rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --show-error \
        --output /tmp/protoc.zip \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
    && \
    unzip /tmp/protoc.zip -d /usr/local && \
    rm /tmp/protoc.zip

RUN python3 -m venv /opt/protobuf-build-venv && \
    python -m pip install --no-cache-dir \
        cmake==4.2.3 \
        ninja==1.13.0 \
        mypy-protobuf==5.1.0

WORKDIR /src/opencda

COPY opencda/CMakeLists.txt ./
COPY opencda/opencda/core/common/communication/CMakeLists.txt \
    opencda/core/common/communication/CMakeLists.txt
COPY opencda/opencda/core/common/communication/messages/ \
    opencda/core/common/communication/messages/

RUN cmake -S . -B /tmp/opencda-protobuf-build -G Ninja \
        -DOPENCDA_BUILD_PROTOBUF=ON \
        -DOPENCDA_BUILD_CUDA=OFF \
    && \
    cmake --build /tmp/opencda-protobuf-build && \
    cmake --install /tmp/opencda-protobuf-build \
        --prefix /opt/opencda-artifacts/protobuf


FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}@${CUDA_DEVEL_DIGEST} AS cuda-builder

ARG CUDA_ARCHITECTURES=86

ENV PATH="/opt/cuda-build-venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.* \
        python3=3.12.* \
        python3-dev=3.12.* \
        python3-venv=3.12.* \
    && \
    rm -rf /var/lib/apt/lists/*

COPY opencda/requirements-cuda.txt /tmp/requirements-cuda.txt

RUN python3 -m venv /opt/cuda-build-venv && \
    python -m pip install --no-cache-dir \
        cmake==4.2.3 \
        ninja==1.13.0 \
        -r /tmp/requirements-cuda.txt

WORKDIR /src/opencda

COPY opencda/CMakeLists.txt ./
COPY opencda/OpenCOOD/opencood/pcdet_utils/ \
    OpenCOOD/opencood/pcdet_utils/

RUN cmake -S . -B /tmp/opencda-cuda-build -G Ninja \
        -DOPENCDA_BUILD_PROTOBUF=OFF \
        -DOPENCDA_BUILD_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DOPENCDA_TORCH_RUNTIME_LIBRARY_DIR=/opt/venv/lib/python3.12/site-packages/torch/lib \
    && \
    cmake --build /tmp/opencda-cuda-build && \
    cmake --install /tmp/opencda-cuda-build \
        --prefix /opt/opencda-artifacts/cuda


FROM runtime-base AS opencda-base

ARG USER=opencda
ARG UID=1000
ARG HOME=/home/${USER}

COPY --from=python-dependencies /opt/venv /opt/venv

WORKDIR ${HOME}/cavise/opencda

COPY --chown=${UID}:${UID} opencda/opencda.py ./
COPY --chown=${UID}:${UID} opencda/opencda/ opencda/
COPY --chown=${UID}:${UID} opencda/OpenCOOD/ OpenCOOD/
COPY --chown=${UID}:${UID} opencda/AIM/ AIM/
COPY opencda/docker/sync-build-artifacts.sh \
    /usr/local/bin/opencda-sync-build-artifacts

RUN chmod 755 /usr/local/bin/opencda-sync-build-artifacts

USER ${USER}

ENTRYPOINT ["/usr/local/bin/opencda-sync-build-artifacts"]
CMD ["bash"]


FROM opencda-base AS opencda-minimal

ENV OPENCDA_NATIVE_COMPONENTS=""


FROM opencda-base AS opencda-protobuf

ENV OPENCDA_NATIVE_COMPONENTS="protobuf"

COPY --from=protobuf-builder /opt/opencda-artifacts/protobuf \
    /opt/opencda-artifacts/protobuf


FROM opencda-base AS opencda-cuda

ENV OPENCDA_NATIVE_COMPONENTS="cuda"

COPY --from=cuda-builder /opt/opencda-artifacts/cuda \
    /opt/opencda-artifacts/cuda


FROM opencda-base AS opencda

ENV OPENCDA_NATIVE_COMPONENTS="protobuf cuda"

COPY --from=protobuf-builder /opt/opencda-artifacts/protobuf \
    /opt/opencda-artifacts/protobuf
COPY --from=cuda-builder /opt/opencda-artifacts/cuda \
    /opt/opencda-artifacts/cuda
