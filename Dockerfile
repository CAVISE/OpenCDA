ARG TAG=base-devel

FROM archlinux:${TAG}

SHELL [ "/bin/bash", "-c"]

ARG USER=opencda
ARG GROUP=opencda
ARG UID=1000
ARG GID=1000
ARG HOME=/home/${USER}

ENV PYENV_ROOT=${HOME}/.pyenv
ENV PYTHONPATH=${HOME}/cavise/opencda
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/opt/cuda/bin:/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/cuda/lib64
ENV CC=/usr/bin/gcc-14
ENV CXX=/usr/bin/g++-14
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_NVCC_FLAGS="--allow-unsupported-compiler"

RUN groupadd -g ${GID} ${GROUP} && \
    useradd -m -u ${UID} -g ${GROUP} -s /bin/bash ${USER} -d ${HOME}

RUN pacman -Syu --noconfirm \
    --disable-download-timeout \
    python3 python-pip pyenv \
    bison openmp pacman-contrib \
    nvidia-utils cuda \
    vulkan-tools \
    libjpeg-turbo libtiff \
    && paccache -r -k 0

USER ${USER}
WORKDIR ${HOME}/cavise/opencda

COPY opencda/.python-version .python-version
COPY opencda/requirements.txt requirements.txt

RUN pyenv install $(pyenv local) && pyenv local && pyenv rehash
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --break-system-packages -r requirements.txt
