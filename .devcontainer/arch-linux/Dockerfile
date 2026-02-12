ARG TAG=base-devel

###############
# Setup stage #
###############

FROM archlinux:${TAG} AS setup

SHELL [ "/bin/bash", "-o", "pipefail", "-c"]

RUN pacman -Syu --noconfirm     \
    --disable-download-timeout  \
    pacman-contrib python3      \
    python-pip pyenv bison git  \
    mesa sdl2 libsm openmp      \
    fontconfig gcc cmake        \
    libjpeg-turbo libtiff nano  \
    unzip xdg-user-dirs whois   \
    ttf-ubuntu-font-family      \
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
RUN cmake -B build . && cmake --build build --parallel "$(nproc --all)"

######################
# Installation stage #
######################

FROM setup AS final

# User name for this container
ARG USER=devcontainer

COPY --from=build /sumo /sumo
RUN cmake --install /sumo/build

RUN groupadd sudo && useradd -l -m -G sudo ${USER}
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER}

WORKDIR /workspaces/opencda
COPY .python-version .python-version
COPY requirements.txt requirements.txt

ENV HOME=/home/${USER}
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN pyenv install "$(pyenv local)" && pyenv local && pyenv rehash
RUN pip3 install --no-cache-dir --break-system-packages \
    -r requirements.txt

ENV PYTHONPATH=/workspaces/opencda
ENV SUMO_HOME=/usr/local/share/sumo
