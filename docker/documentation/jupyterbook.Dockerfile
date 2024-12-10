ARG PYTHON_VERSION=3.9
ARG CONTEXT_DIR=.
ARG HOME_DIR=.
ARG CONTENT_DIR=omniverse
ARG VENV_DIR=/opt
ARG VENV_NAME=venv

FROM python:${PYTHON_VERSION}-bullseye as builder

ARG CONTEXT_DIR
ARG HOME_DIR
ARG CONTENT_DIR
ARG VENV_DIR
ARG VENV_NAME

WORKDIR ${HOME_DIR}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_DIR}/${VENV_NAME}
ENV PATH="${VENV_DIR}/${VENV_NAME}/bin:$PATH"

ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_DEV=requirements-dev.txt
COPY ./${CONTEXT_DIR}/${REQUIREMENTS} .
COPY ./${CONTEXT_DIR}/${REQUIREMENTS_DEV} .

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS} && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS_DEV}

COPY ./${CONTEXT_DIR}/${CONTENT_DIR} ${HOME_DIR}/${CONTENT_DIR}

RUN echo "Listing contents of HOME_DIR (${HOME_DIR}):" && \
    ls -l ${HOME_DIR} && \
    echo "Listing contents of CONTENT_DIR (${HOME_DIR}/${CONTENT_DIR}):" && \
    ls -l ${HOME_DIR}/${CONTENT_DIR} && \
    echo "Current working directory:" && \
    pwd

RUN jupyter-book build ${CONTENT_DIR}

FROM nginx:alpine as runner

ARG HOME_DIR
ARG CONTENT_DIR

COPY --from=builder ${HOME_DIR}/${CONTENT_DIR}/_build/html /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
