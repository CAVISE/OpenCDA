ARG TAG=base-devel

FROM archlinux:${TAG}

SHELL [ "/bin/bash", "-c"]

ARG USER=opencda
ARG GROUP=opencda
ARG UID=1000
ARG GID=1000
ARG HOME=/home/${USER}

ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
ENV PYTHONPATH=/home/${USER}/cavise/opencda
ENV PATH=/opt/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

RUN groupadd -g ${GID} ${GROUP} && \
    useradd -m -u ${UID} -g ${GROUP} -s /bin/bash ${USER} -d ${HOME}

RUN pacman -Syu --noconfirm      \
    --disable-download-timeout   \
    python3 python-pip pyenv     \
    python-pip pyenv bison       \
    libsm openmp vim nano        \
    xdg-user-dirs pacman-contrib \
    xorg nvidia-utils            \
    vulkan-tools cuda            \
    vulkan-icd-loader            \
    libjpeg-turbo libtiff        \
    && paccache -r -k 0

# mesa fontconfig ttf-ubuntu-font-family sdl2

USER ${USER}
WORKDIR ${HOME}/cavise/opencda

COPY opencda/.python-version .python-version
COPY opencda/requirements.txt requirements.txt

RUN pyenv install $(pyenv local) && pyenv local && pyenv rehash
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
