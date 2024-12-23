FROM archlinux:base-devel-20240804.0.251467 AS carla

SHELL [ "/bin/bash", "-c"]

# if sumo needed
ARG SUMO=true
# user name to use
ARG USER=opencda

WORKDIR /cavise

RUN pacman -Syu --noconfirm pacman-contrib
RUN pacman -S --noconfirm python3 python-pip pyenv bison git gcc cmake
RUN pacman -S --noconfirm xorg nvidia-utils mesa sdl2 libsm openmp qt5-base \
        fontconfig libjpeg-turbo libtiff nano unzip xdg-user-dirs whois \
        vulkan-tools vulkan-icd-loader

########################################################
# Install SUMO
########################################################
RUN if [ "${SUMO}" = "true" ]; then                                                         \
        # reference https://sumo.dlr.de/docs/Installing/Linux_Build.html
        pacman -S --noconfirm xerces-c fox gdal proj gl2ps jre17-openjdk                    \
            swig maven eigen &&                                                             \
        git clone --recurse --depth 1 https://github.com/eclipse-sumo/sumo &&               \
        cd sumo && cmake -B build . && cmake --build build -j$(nproc --all) &&              \
        cmake --install build                                                               \ 
    ; else                                                                                  \
        echo "Installation without SUMO"                                                    \
    ; fi
ENV SUMO_HOME="/usr/local/share/sumo"

# reference: https://github.com/carla-simulator/carla/issues/5791
RUN cd /usr/lib && ln -s libomp.so libomp.so.5
# same as above
RUN cd /usr/lib && ln -s libtiff.so libtiff.so.5

# For manual control
RUN pacman -S --noconfirm ttf-ubuntu-font-family

RUN useradd -m -s /bin/bash ${USER}
USER ${USER}

# You should mount opencda dynamically
ADD . /cavise

WORKDIR /cavise

ENV HOME=/home/${USER}
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN pyenv install $(pyenv local)
RUN pyenv local
RUN pyenv rehash

WORKDIR /cavise/opencda

RUN pip3 install --break-system-packages -r requirements.txt
RUN pip3 install --break-system-packages -r requirements_ci.txt
RUN pip3 install --break-system-packages -qr https://raw.githubusercontent.com/ultralytics/yolov5/v3.0/requirements.txt
RUN pip3 install --break-system-packages ultralytics==8.0.145
RUN pip3 install --break-system-packages -r requirements_cavise.txt

RUN if [ "${SUMO}" = "true" ]; then pip3 install --break-system-packages traci; fi

ENV PYTHONPATH=/cavise
ENV CAVISE_ROOT_DIR=/cavise

CMD ["echo", "'run this interactively'"]