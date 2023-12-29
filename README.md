# Omniverse

![Python version](https://img.shields.io/badge/Python-3.9-3776AB)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/gaohongnan.svg?style=social&label=Follow%20%40gaohongnan)](https://twitter.com/gaohongnan)
[![LinkedIn](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![Continuous Integration Checks on Omnivault.](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml/badge.svg)](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml)

-   [Omniverse](#omniverse)
    -   [Building and Running the Jupyter Book Docker Image](#building-and-running-the-jupyter-book-docker-image)
        -   [Building the Docker Image](#building-the-docker-image)
        -   [Running the Docker Container](#running-the-docker-container)
        -   [Stopping the Docker Container](#stopping-the-docker-container)
        -   [Further Enhancements](#further-enhancements)
        -   [References and Further Readings](#references-and-further-readings)
    -   [Release using GitHub Actions CI/CD Workflows](#release-using-github-actions-cicd-workflows)
        -   [Semantic Versioning](#semantic-versioning)
            -   [Format](#format)
            -   [Example Versioning](#example-versioning)
            -   [Pre-release and Build Metadata](#pre-release-and-build-metadata)
        -   [Release using GitHub Actions CI/CD Workflows](#release-using-github-actions-cicd-workflows-1)
        -   [Example Workflow](#example-workflow)
        -   [References and Further Readings](#references-and-further-readings-1)

🌌 Omniverse: A cosmic collection of machine learning, deep learning, data
science, math, and software engineering explorations. Dive into the universe of
knowledge! 🚀 To create a detailed Markdown section in the `README.md` file for
instructing users on how to build and run the Dockerfile
`jupyterbook.Dockerfile`, you should include steps that cover prerequisites,
building the Docker image, tagging it with the Git commit ID, and running the
container. Additionally, to avoid hardcoding variables in the Docker build/run
commands, you can use shell variables and command substitutions.

## Building and Running the Jupyter Book Docker Image

This section provides detailed instructions on how to build the Dockerfile
(`docker/documentation/jupyterbook.Dockerfile`) and run the Docker image. The
image provides a containerized environment for building and serving the Jupyter
Book website.

> [!WARNING] This section is only tested on macOS Ventura 13.4.1.

First, ensure you are in the root directory of the repository, if not, change
directories to the root directory:

```bash
~/ $ cd <path/to/omniverse>
```

Replace `<path/to/omniverse>` with the actual path to your repository's root
directory.

### Building the Docker Image

1. **Set Environment Variables**: Set the `GIT_COMMIT_HASH`, `IMAGE_NAME`, and
   `IMAGE_TAG` environment variables. These will be used to tag your Docker
   image uniquely.

    ```bash
    ~/omniverse $ export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
    ~/omniverse $ export IMAGE_NAME=omniverse
    ~/omniverse $ export IMAGE_TAG=$GIT_COMMIT_HASH
    ```

2. **Build the Image**: Execute the following Docker command to build the image,
   specifying the Dockerfile path and assigning the tag based on the previously
   set environment variables.

    ```bash
    ~/omniverse $ docker build \
    >   --file docker/documentation/jupyterbook.Dockerfile \
    >   --tag $IMAGE_NAME:$IMAGE_TAG \
    >   .
    ```

### Running the Docker Container

To run the Docker container:

```bash
~/omniverse $ docker run \
>   --publish 80:80 \
>   $IMAGE_NAME:$IMAGE_TAG
```

This command will start a container from the built image, mapping port 80 of the
**container** to port 80 on the **host** machine. The website should now be
accessible at `http://localhost:80`.

### Stopping the Docker Container

To stop the Docker container:

```bash
~/omniverse $ export CONTAINER_ID=$(docker ps --filter ancestor=$IMAGE_NAME:$IMAGE_TAG --format "{{.ID}}")
~/omniverse $ docker stop $CONTAINER_ID
```

### Further Enhancements

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

### References and Further Readings

-   [How to run Nginx within a Docker container without halting?](https://stackoverflow.com/questions/18861300/how-to-run-nginx-within-a-docker-container-without-halting)

## Release using GitHub Actions CI/CD Workflows

### Semantic Versioning

The conventional way to name software versions is by following
[**Semantic Versioning**](https://semver.org/) (SemVer). Semantic Versioning is
a widely adopted system for versioning software in a way that conveys meaning
about the underlying changes. It helps in managing dependencies and avoiding
compatibility issues.

#### Format

A Semantic Version number is typically formatted as `MAJOR.MINOR.PATCH`, where:

1. **MAJOR version**:

    - Incremented for incompatible API changes or major changes in
      functionality.
    - Indicates that the new version might not be backward compatible with
      previous major versions.

2. **MINOR version**:

    - Incremented for adding functionality in a backward-compatible manner.
    - Indicates new features or improvements which do not break existing
      functionalities.

3. **PATCH version**:
    - Incremented for backward-compatible bug fixes.
    - Focuses on resolving bugs and issues without adding new features or
      breaking existing functionality.

#### Example Versioning

-   `1.0.0`: The first stable release of a software.
-   `1.0.1`: A subsequent release that includes bug fixes but no new features.
-   `1.1.0`: A release that introduces new features but is still backward
    compatible with the `1.0.x` series.
-   `2.0.0`: A release that makes changes significant enough to potentially
    break compatibility with the `1.x.x` series.

#### Pre-release and Build Metadata

Semantic Versioning also supports additional labels for pre-release and build
metadata:

-   Pre-release versions can be denoted with a hyphen and additional identifiers
    (e.g., `2.0.0-alpha`, `2.0.0-beta.1`).
-   Build metadata can be appended using a plus sign and additional identifiers
    (e.g., `2.0.0+build.20230315`).

### Release using GitHub Actions CI/CD Workflows

Follow
[the guide here](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
for detailed instructions on how to release a Python package using GitHub
Actions CI/CD workflows.

One thing that is not mentioned in the guide, but is a good practice, is to do
all the necessary pre-release checks before releasing your package, that
includes all the pre-merge checks, and additional checks such as running tests,
linting, and building the documentation. This ensures that the release is of
high quality and is ready to be used by others.

Furthermore, we add a `release-docker` workflow to build and publish a Docker
image to Docker Hub. The workflow is triggered when a new release is published
to PyPI. This approach is inspired by
[langchain's workflow](https://github.com/langchain-ai/langchain/blob/master/.github/workflows/langchain_release.yml)
for publishing a Docker image via GitHub Actions. The rationale behind
incorporating a Docker release alongside the PyPI release is to ensure the
package `omniverse` is able to be imported and used across different platforms.

### Example Workflow

Update the `version` field in `pyproject.toml` to the new version, and commit
the changes to the `main` branch (or any other branch that satisfies the
`on.push.branches` condition in the workflow).

```bash
git commit -m "bump version to 0.0.10"
git tag -a v0.0.10 -m "Release version 0.0.10"
git push origin main
git push origin v0.0.10
```

Then the workflow will be triggered, and the package will be published to PyPI.
It is worth noting that we will do a pre-release check before publishing the
package.

### References and Further Readings

-   [Packaging Projects - Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/)
-   [Publishing Package Distribution Releases Using GitHub Actions CI/CD Workflows - Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
-   [Publishing Python Packages](https://carpentries-incubator.github.io/python_packaging/instructor/05-publishing.html)
-   [Publishing Python Packages from GitHub Actions](https://www.seanh.cc/2022/05/21/publishing-python-packages-from-github-actions/)
