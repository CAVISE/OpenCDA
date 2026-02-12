# TODO: Check why opencda mouts to devcontainer instead of just OpenCOOD
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed AS opencda
SHELL [ "/bin/bash", "-c"]
ARG USER=devcontainer
ARG UID=1000
ARG HOME=/home/${USER}
ARG WORKSPACE=/workspaces/opencda

ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_NVCC_FLAGS="--allow-unsupported-compiler"
ENV PYTHONPATH="/workspaces/opencda/OpenCOOD"

RUN userdel -r ubuntu && useradd -l -m -u ${UID} -s /bin/bash ${USER} -d ${HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip=24.0+dfsg-1ubuntu1.3 \
        python3-dev=3.12.3-0ubuntu2.1 \
        libgl1=1.7.0-1build1 \
        libglib2.0-0=2.80.0-6ubuntu1.2 \
        libsm6=2:1.2.3-1build3 \
        libxext6=2:1.3.4-1build2 \
        libxrender1=1:0.9.10-1.1build1 \
        && \
        apt-get autoremove -y && \
        rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python

USER ${USER}
WORKDIR ${HOME}

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip==24.0 && \
    python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

CMD ["/bin/bash", "-c", "python3 opencood/utils/setup.py build_ext --inplace && python opencood/pcdet_utils/setup.py build_ext --inplace"]
