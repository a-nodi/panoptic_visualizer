FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN sed -i 's|http://archive.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list
RUN sed -i 's|http://security.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV TZ=Asia/Seoul

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wget unzip zip
RUN apt-get update --fix-missing

RUN apt-get install -y \
    build-essential cmake libtool autoconf \
    bzip2 curl git tty-clock vim tmux dpkg \
    htop kmod gcc-10 g++-10 clang

RUN apt-get update --fix-missing
RUN apt-get upgrade -y

RUN apt-get install sudo
RUN apt-get update
RUN apt-get autoremove

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN wget https://github.com/isl-org/Open3D/archive/refs/tags/v0.16.0.zip -O ~/Open3D.zip
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git

RUN sh ~/miniconda.sh -b

RUN sudo apt-get install -y libosmesa6-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libglib2.0-0
WORKDIR /root
RUN unzip Open3D.zip
WORKDIR /root/Open3D-0.18.0
RUN mkdir build

WORKDIR /root

COPY conda_open3d_install.sh .
RUN chmod +x conda_open3d_install.sh
RUN bash conda_open3d_install.sh

COPY conda_pytorch_install.sh .
RUN chmod +x conda_pytorch_install.sh
RUN bash conda_pytorch_install.sh

COPY conda_minkowskiengine_install.sh .
RUN chmod +x conda_minkowskiengine_install.sh
RUN mv ../MinkowskiEngine ./MinkowskiEngine
RUN bash conda_minkowskiengine_install.sh

COPY conda_nuscenes_devkit_install.sh .
RUN chmod +x conda_nuscenes_devkit_install.sh
RUN bash conda_nuscenes_devkit_install.sh

COPY requirements.txt .
COPY conda_requirements_install.sh .
RUN chmod +x conda_requirements_install.sh
RUN bash conda_requirements_install.sh

RUN rm /root/Open3D.zip
RUN rm /root/miniconda.sh
RUN rm /root/conda_open3d_install.sh
RUN rm /root/conda_pytorch_install.sh
RUN rm /root/conda_minkowskiengine_install.sh
RUN rm /root/conda_nuscenes_devkit_install.sh
RUN rm /root/conda_requirements_install.sh
RUN rm /root/requirements.txt
