FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"

ARG NON_ROOT_USER="gaohn"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"

ARG REPO_DIR="."
ARG CONDA_ENV_NAME="omniverse"

# Miniconda arguments
ARG CONDA_HOME="/miniconda3"
ARG CONDA_BIN="${CONDA_HOME}/bin/conda"
ARG CONDA_VER="py310_23.5.2-0"
ARG CONDA_ARCH="Linux-x86_64"
ARG MINICONDA_SH="Miniconda3-${CONDA_VER}-${CONDA_ARCH}.sh"

RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER} && \
    mkdir -p ${CONDA_HOME} && \
    chown -R ${NON_ROOT_USER}:${NON_ROOT_GID} ${CONDA_HOME}

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get -y install bzip2 curl wget gcc rsync git vim locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean

ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} omniverse

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/${MINICONDA_SH} && \
    chmod +x ${MINICONDA_SH} && \
    ./${MINICONDA_SH} -u -b -p ${CONDA_HOME} && \
    rm ${MINICONDA_SH}
ENV PATH ${CONDA_HOME}/bin:${HOME_DIR}/.local/bin:$PATH

# Ensure base conda environment is activated by default
RUN echo "source ${CONDA_HOME}/bin/activate" >> "${HOME_DIR}/.bashrc"

# Use ENTRYPOINT to activate the conda base environment upon startup
ENTRYPOINT [ "/bin/bash", "-c", "source ${CONDA_HOME}/bin/activate && exec bash" ]
