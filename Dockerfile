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
        vulkan-tools vulkan-icd-loader ttf-ubuntu-font-family cuda && \
    pacman -Sc --noconfirm && \
    rm -rf /var/cache/pacman/pkg/* /tmp/*

ENV PATH=/opt/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH \
    SUMO_HOME="/usr/local/share/sumo" \
    PYTHONPATH=/cavise \
    CAVISE_ROOT_DIR=/cavise

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
        cmake --install build && \
        rm -rf /cavise/sumo /var/cache/pacman/pkg/* /tmp/*                                              \ 
    ; else                                                                                              \
        echo "Installation without SUMO"                                                                \
    ; fi

# reference: https://github.com/carla-simulator/carla/issues/5791
RUN cd /usr/lib && ln -s libomp.so libomp.so.5 && \
    cd /usr/lib && ln -s libtiff.so libtiff.so.5

RUN useradd -m -s /bin/bash ${USER}
USER ${USER}

# You should mount opencda dynamically
ADD opencda /cavise/opencda

WORKDIR /cavise/opencda

ENV HOME=/home/${USER}
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN pyenv install $(pyenv local) &&\
    pyenv local && \
    pyenv rehash
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt -r requirements_ci.txt -r requirements_cavise.txt
#RUN pip3 install --break-system-packages -qr https://raw.githubusercontent.com/ultralytics/yolov5/v3.0/requirements.txt
#RUN pip3 install --break-system-packages ultralytics==8.0.145

RUN if [ "${SUMO}" = "true" ]; then pip3 install --no-cache-dir --break-system-packages traci; fi

CMD ["echo", "'run this interactively'"]