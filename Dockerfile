FROM archlinux:base-devel AS carla

SHELL [ "/bin/bash", "-c"]

# if sumo needed
ARG SUMO=true
ARG SUMO_VERSION=v1_21_0
# user name to use
ARG USER=opencda

WORKDIR /cavise

RUN pacman -Syu --noconfirm pacman-contrib && \
    pacman -S --noconfirm python3 python-pip pyenv bison git gcc cmake && \
    pacman -S --noconfirm xorg nvidia-utils mesa sdl2 libsm openmp qt5-base \
        fontconfig libjpeg-turbo libtiff nano unzip xdg-user-dirs whois \
        vulkan-tools vulkan-icd-loader ttf-ubuntu-font-family cuda

ENV PATH=/opt/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Manual control requires ttf-ubuntu-font-family

########################################################
# Install SUMO
########################################################
RUN if [ "${SUMO}" = "true" ]; then                                                                     \
        # reference https://sumo.dlr.de/docs/Installing/Linux_Build.html
        pacman -S --noconfirm xerces-c fox gdal proj gl2ps jre17-openjdk                                \
            swig maven eigen &&                                                                         \
        git clone --recurse --depth 1 --branch ${SUMO_VERSION} https://github.com/eclipse-sumo/sumo &&  \
        cd sumo && cmake -B build . && cmake --build build -j$(nproc --all) &&                          \
        cmake --install build                                                                           \ 
    ; else                                                                                              \
        echo "Installation without SUMO"                                                                \
    ; fi
ENV SUMO_HOME="/usr/local/share/sumo"

# reference: https://github.com/carla-simulator/carla/issues/5791
RUN cd /usr/lib && ln -s libomp.so libomp.so.5 && \
    cd /usr/lib && ln -s libtiff.so libtiff.so.5

RUN useradd -m -s /bin/bash ${USER}
USER ${USER}

# You should mount opencda dynamically
ADD . /cavise

WORKDIR /cavise/opencda

ENV HOME=/home/${USER}
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN pyenv install $(pyenv local) &&\
    pyenv local && \
    pyenv rehash

RUN pip3 install --break-system-packages -r requirements.txt && \
    pip3 install --break-system-packages -r requirements_ci.txt && \
    pip3 install --break-system-packages -r requirements_cavise.txt
#RUN pip3 install --break-system-packages -qr https://raw.githubusercontent.com/ultralytics/yolov5/v3.0/requirements.txt
#RUN pip3 install --break-system-packages ultralytics==8.0.145

RUN if [ "${SUMO}" = "true" ]; then pip3 install --break-system-packages traci; fi

ENV PYTHONPATH=/cavise
ENV CAVISE_ROOT_DIR=/cavise

CMD ["echo", "'run this interactively'"]