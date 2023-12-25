# Omniverse

![Python version](https://img.shields.io/badge/Python-3.9-3776AB)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/gaohongnan.svg?style=social&label=Follow%20%40gaohongnan)](https://twitter.com/gaohongnan)
[![LinkedIn](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![Continuous Integration Checks on Omnivault.](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml/badge.svg)](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml)

🌌 Omniverse: A cosmic collection of machine learning, deep learning, data
science, math, and software engineering explorations. Dive into the universe of
knowledge! 🚀 To create a detailed Markdown section in the `README.md` file for
instructing users on how to build and run the Dockerfile
`jupyterbook.Dockerfile`, you should include steps that cover prerequisites,
building the Docker image, tagging it with the Git commit ID, and running the
container. Additionally, to avoid hardcoding variables in the Docker build/run
commands, you can use shell variables and command substitutions.

## Building and Running the Jupyter Book Docker Image

This section provides detailed instructions on how to build and run the
`scripts/docker/documentation/jupyterbook.Dockerfile` Docker image. The image
provides a containerized environment for building and serving the Jupyter Book
website.

First, ensure you are in the root directory of the repository, if not, change
directories to the root directory:

```bash
cd <path/to/omniverse>
```

### Building the Docker Image

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
export IMAGE_NAME=omniverse
export IMAGE_TAG=$GIT_COMMIT_HASH
docker build \
  --file scripts/docker/documentation/jupyterbook.Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .
```

### Running the Docker Container

To run the Docker container:

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
export IMAGE_NAME=omniverse
export IMAGE_TAG=$GIT_COMMIT_HASH
docker run \
  --publish 80:80 \
  $IMAGE_NAME:$IMAGE_TAG
```

This command will start a container from the built image, mapping port 80 of the
**container** to port 80 on the **host** machine. The website should now be
accessible at `http://localhost:80`.

### References and Further Readings

This is a relatively simple Dockerfile. Further enhancements can include, but
not limited to:

-   Add entrypoint script to start the Jupyter Book server.
-   Use Docker Compose to manage multiple containers, for example, a container
    for development of the content and a container for serving the website.
-   Current Docker image is used primarily for serving, and users may find it
    hard to directly develop **inside** the container. A better approach is to
    use a Docker image for development, and mount the source code directory to
    the container. This way, users can develop on their host machine, and the
    changes will be reflected in the container.

-   [How to run Nginx within a Docker container without halting?](https://stackoverflow.com/questions/18861300/how-to-run-nginx-within-a-docker-container-without-halting)
