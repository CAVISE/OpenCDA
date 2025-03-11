ARG TAG=base-devel

###############
# Setup stage #
###############

FROM archlinux:${TAG} AS setup

SHELL [ "/bin/bash", "-c"]

RUN pacman -Syu --noconfirm     \
    --disable-download-timeout  \
    pacman-contrib python3      \
    python-pip pyenv bison git  \
    mesa sdl2 libsm openmp      \
    fontconfig gcc cmake        \
    libjpeg-turbo libtiff nano  \ 
    unzip xdg-user-dirs whois   \
    ttf-ubuntu-font-family      \
    xorg nvidia-utils mesa      \
    vulkan-tools cuda           \
    vulkan-icd-loader           \
    # Sumo build deps           
    xerces-c fox gdal proj      \
    gl2ps jre17-openjdk         \
    swig maven eigen            \
    && paccache -r -k 0

###############
# Build stage #
###############

FROM setup AS build

# sumo version
ARG SUMO_TAG=v1_21_0

RUN git clone --recurse --depth 1 --branch ${SUMO_TAG} https://github.com/eclipse-sumo/sumo
WORKDIR /sumo
RUN cmake -B build . && cmake --build build --parallel $(nproc --all)

######################
# Installation stage #
######################

FROM setup AS final

# User name for this container
ARG USER=opencda

COPY --from=build /sumo /sumo
RUN cmake --install /sumo/build

RUN groupadd sudo && useradd -m -G sudo ${USER}
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/${USER}

WORKDIR /home/${USER}/opencda
COPY opencda/.python-version .python-version
COPY opencda/requirements_ci.txt requirements_ci.txt
COPY opencda/requirements.txt requirements.txt
COPY opencda/requirements_cavise.txt requirements_cavise.txt

ENV HOME=/home/${USER}
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN pyenv install $(pyenv local) && pyenv local && pyenv rehash
RUN pip3 install --no-cache-dir --break-system-packages \
    -r requirements.txt -r requirements_ci.txt -r requirements_cavise.txt traci

ENV PYTHONPATH=/home/${USER}/opencda
ENV SUMO_HOME=/usr/local/share/sumo

ENV PATH=/opt/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH