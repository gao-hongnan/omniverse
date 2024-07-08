# Omniverse

![Python version](https://img.shields.io/badge/Python-3.9-3776AB)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/gaohongnan.svg?style=social&label=Follow%20%40gaohongnan)](https://twitter.com/gaohongnan)
[![LinkedIn](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![Continuous Integration Checks on Omnivault.](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml/badge.svg)](https://github.com/gao-hongnan/omniverse/actions/workflows/ci_omnivault.yaml)

- [Omniverse](#omniverse)
    - [Blogs](#blogs)
    - [Implementation of Decoder](#implementation-of-decoder)
    - [Building and Running NVIDIA Docker Image](#building-and-running-nvidia-docker-image)
    - [Building and Running Jupyter Book Docker Image](#building-and-running-jupyter-book-docker-image)
        - [Building the Docker Image](#building-the-docker-image)
        - [Running the Docker Container](#running-the-docker-container)
        - [Stopping the Docker Container](#stopping-the-docker-container)
        - [Further Enhancements](#further-enhancements)
        - [References and Further Readings](#references-and-further-readings)
    - [Release using GitHub Actions CI/CD Workflows](#release-using-github-actions-cicd-workflows)
        - [Semantic Versioning](#semantic-versioning)
            - [Format](#format)
            - [Example Versioning](#example-versioning)
            - [Pre-release and Build Metadata](#pre-release-and-build-metadata)
        - [Release using GitHub Actions CI/CD Workflows](#release-using-github-actions-cicd-workflows-1)
        - [Example Workflow](#example-workflow)
    - [Custom Domain for GitHub Pages](#custom-domain-for-github-pages)
    - [How to Index Jupyter Book?](#how-to-index-jupyter-book)
        - [References and Further Readings](#references-and-further-readings-1)

🌌 Omniverse: A cosmic collection of machine learning, deep learning, data
science, math, and software engineering explorations. Dive into the universe of
knowledge! 🚀

## Blogs

-   https://gao-hongnan.github.io/gaohn-galaxy
-   https://www.gaohongnan.com (migrating from the previous blog)

The tiered structure of the blogs is as follows:

-   ![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red) represents raw,
    unfiltered and maybe wrong thoughts.
-   ![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange) represents
    organized thoughts, but still chaotic and potentially incorrect.
-   ![Tag](https://img.shields.io/badge/Tag-Structured_Musings-purple)
    represents structured knowledge, but not necessarily correct.
-   ![Tag](https://img.shields.io/badge/Tag-Polished_Insights-blue) represents
    refined knowledge, and is polished and correct.
-   ![Tag](https://img.shields.io/badge/Tag-Mastery-green) represents mastery of
    the subject.

## Implementation of Decoder

-   https://github.com/gao-hongnan/omniverse/tree/main/omnivault/transformer

## Building and Running NVIDIA Docker Image

Currently our `.github/workflows/nvidia-docker.yaml` workflow is used to build
and push the Docker image to Docker Hub. To use it, follow the below steps.

```bash
cd </path/to/project> # i.e. go to omniverse dir first so we can call pwd
chmod -R 777 $PWD # need for mkdir etc
docker run --gpus all -it --user 2222:2222 --shm-size=16g -v $PWD:/workspace gaohn/omniverse-nvidia:6140759e
```

The `-shm-size` is needed because if your virtual machine has say, 8 CPU cores,
then you would likely use 8 workers in the dataloading, and you would require
more shared memory.

## Building and Running Jupyter Book Docker Image

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
VERSION="0.0.57"
git commit -am "cicd: bump version to #$VERSION [#38]." && \
git tag -a v$VERSION  -m "Release version $VERSION" && \
git push && \
git push origin v$VERSION
```

Then the workflow will be triggered, and the package will be published to PyPI.
It is worth noting that we will do a pre-release check before publishing the
package.

## Custom Domain for GitHub Pages

To use a custom domain with GitHub Pages and with Jupyter Book, we would need to
follow the instructions given
[here](https://jupyterbook.org/en/stable/publish/gh-pages.html#use-a-custom-domain-with-github-pages).

1. **Add Custom Domain to GitHub Pages Settings**:

    - Go to your GitHub repository.
    - Click on "Settings".
    - Scroll down to the "GitHub Pages" section.
    - In the "Custom domain" box, enter your custom domain (e.g.,
      `gaohongnan.com`) and save.
    - You might see the "improperly configured" error, which is expected at this
      stage since the DNS hasn't been set up yet.

    > Make sure you add your custom domain to your GitHub Pages site before
    > configuring your custom domain with your DNS provider. Configuring your
    > custom domain with your DNS provider without adding your custom domain to
    > GitHub could result in someone else being able to host a site on one of
    > your subdomains. From GitHub
    > [documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#about-custom-domain-configuration)

2. **Modify DNS Settings at Domain Registrar**:

    - Head over to your domain registrar.
    - Configure the DNS settings:
        - For an apex domain: Set up the **A records**.
        - For a `www` subdomain: Set up the **CNAME record** pointing to your
          GitHub Pages URL.

3. **Wait and Check**:

    - Now, you'll need to wait for DNS propagation. This can take some time.
    - After a while (it could be a few hours), return to your GitHub Pages
      settings. The error should resolve itself once the DNS has fully
      propagated and GitHub can detect the correct settings.

4. **Enforce HTTPS**:
    - Once the error is gone, you can then opt to "Enforce HTTPS" for added
      security.

In essence, you temporarily accept the error message in your GitHub Pages
settings after adding the custom domain. After you've configured the DNS
settings at your domain registrar and they've propagated, the error in GitHub
Pages settings should clear up.

The main goal of GitHub's recommendation is to make sure you've shown intent to
use the domain with GitHub Pages before setting it up with your DNS provider, to
prevent potential subdomain takeovers. By adding the custom domain in the
repository settings (even if it throws an error initially), you've asserted this
intent.

## How to Index Jupyter Book?

-   [Indexing on search engines](https://github.com/executablebooks/jupyter-book/issues/1934)
-   [Generate sitemap.xml for SEO](https://github.com/executablebooks/jupyter-book/issues/880)

### References and Further Readings

-   [Packaging Projects - Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/)
-   [Publishing Package Distribution Releases Using GitHub Actions CI/CD Workflows - Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
-   [Publishing Python Packages](https://carpentries-incubator.github.io/python_packaging/instructor/05-publishing.html)
-   [Publishing Python Packages from GitHub Actions](https://www.seanh.cc/2022/05/21/publishing-python-packages-from-github-actions/)
