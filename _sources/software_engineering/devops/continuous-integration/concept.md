# Continuous Integration (CI) Workflow

```{contents}
:local:
```

Continuous Integration (CI) is a software development practice that focuses on
frequently integrating code changes from multiple developers into a shared
repository. CI aims to detect integration issues early and ensure that the
software remains in a releasable state at all times.

A CI workflow refers to the series of steps and automated processes involved in
the continuous integration of code changes. It typically involves building and
testing the software, often in an automated and reproducible manner.

The primary goal of a CI workflow is to identify any issues or conflicts that
may arise when integrating code changes. By continuously integrating code,
developers can catch and fix problems early, reducing the chances of introducing
bugs or breaking existing functionality.

CI workflows usually include steps such as code compilation, running tests,
generating documentation, and deploying the application to test environments.
These steps are often automated, triggered by events such as code pushes or pull
requests.

Popular CI tools like Jenkins, Travis CI, and GitHub Actions provide mechanisms
to define and execute CI workflows. These tools integrate with version control
systems and offer a range of features, including customizable build and test
environments, notifications, and integration with other development tools.

## Phase 1. Planning

The Planning stage involves defining what needs to be built or developed. It's a
crucial phase where project managers, developers, and stakeholders come together
to identify requirements, set goals, establish timelines, and plan the resources
needed for the project. This stage often involves using project tracking tools
and Agile methodologies like scrum or kanban to organize tasks, sprints, and
priorities.

Common tech stack includes:

-   [Jira](https://www.atlassian.com/software/jira): A popular project
    management tool that helps teams plan, track, and manage agile software
    development projects.
-   [Confluence](https://www.atlassian.com/software/confluence): A team
    collaboration tool that helps teams create, share, and collaborate on
    projects.

## Phase 2. Development

Coding or development is where the actual software creation takes place.
Developers write code to implement the planned features and functionalities,
adhering to coding standards and practices. **Version control systems**, such as
Git, play an important role in this stage, enabling developers to collaborate on
code, manage changes, and maintain a history of the project's development.

### Set Up Main Directory in Integrated Development Environment (IDE)

Let us assume that we are residing in our root folder `~/` and we want to create
a new project called **yolo** in Microsoft Visual Studio Code, we can do as
follows:

```bash title="creating main directory" linenums="1"
~/      $ mkdir yolo && cd yolo
~/yolo  $ code .                 # (1)
```

If you are cloning a repository to your local folder **yolo**, you can also do:

```bash title="cloning repository" linenums="1"
~/yolo $ git clone git@github.com:<username>/<repo-name>.git .
```

where `.` means cloning to the current directory.

### README, LICENSE and CONTRIBUTING

#### 1. README

The `README.md` file serves as the front page of your repository. It should
provide all the necessary information about the project, including:

-   **Project Name and Description**: Clearly state what your project does and
    why it exists.
-   **Installation Instructions**: Provide a step-by-step guide on how to get
    your project running on a user's local environment.
-   **Usage Guide**: Explain how to use your project, including examples of
    common use cases.
-   **Contributing**: Link to your `CONTRIBUTING.md` file and invite others to
    contribute to the project.
-   **License**: Mention the type of license the project is under, with a link
    to the full `LICENSE` file.
-   **Contact Information**: Offer a way for users to ask questions or get in
    touch.

We can create a `README.md` file to describe the project using the following
command:

```bash title="README.md" linenums="1"
~/yolo $ touch README.md
```

#### 2. LICENSE

The `LICENSE` file is critical as it defines how others can legally use, modify,
and distribute your project. If you’re unsure which license to use,
[choosealicense.com](https://choosealicense.com/) can help you decide.

```bash
~/yolo $ touch LICENSE
```

After creating the file, you should fill it with the text of the license you've
chosen. This could be the MIT License, GNU General Public License (GPL), Apache
License 2.0, etc.

#### 3. CONTRIBUTING

`CONTRIBUTING.md` outlines guidelines for contributing to your project. This
might include:

-   **How to File an Issue**: Instructions for reporting bugs or suggesting
    enhancements.
-   **How to Submit a Pull Request (PR)**: Guidelines on the process for
    submitting a PR, including coding standards, test instructions, etc.
-   **Community and Behavioral Expectations**: Information on the code of
    conduct and the expectations for community interactions.

```bash
~/yolo $ touch CONTRIBUTING.md
```

### Version Control

Ve

#### Initial Configuration

Before you start using Git, you should configure your global username and email
associated with your commits. This information identifies the author of the
changes and is important for collaboration.

```bash
git config --global user.name <your-name>
git config --global user.email <your-email>
```

You should necessarily replace `<your-name>` and `<your-email>` with your actual
name and email address. In particular, having the correct email address is
important as it is used to link your commits to your GitHub account.

#### Setting Up a New Repository

If you're starting a new project in a local directory (e.g., `~/yolo`), you'll
follow these steps to initialize it as a Git repository, stage your files, and
make your first commit.

1. **Create a `.gitignore` File**: This file tells Git which files or
   directories to ignore in your project, like build directories or system
   files.

    ```bash
    ~/yolo $ touch .gitignore
    ```

    Populate `.gitignore` with patterns to ignore. For example:

    ```plaintext
    .DS_Store
    __pycache__/
    env/
    ```

    which can be done using the following command:

    ```bash
    ~/yolo $ cat > .abc <<EOF
    .DS_Store
    __pycache__/
    env/
    EOF
    ```

2. **Initialize the Repository**:

    ```bash
    ~/yolo $ git init
    ```

    This command creates a new Git repository locally.

3. **Add Files to the Repository**:

    ```bash
    ~/yolo $ git add .
    ```

    This adds all files in the directory (except those listed in `.gitignore`)
    to the staging area, preparing them for commit.

4. **Commit the Changes**:

    ```bash
    ~/yolo $ git commit -m "Initial commit"
    ```

    This captures a snapshot of the project's currently staged changes.

#### Connecting to a Remote Repository

After initializing your local repository, the next step is to link it with a
remote repository. This allows you to push your changes to a server, making
collaboration and backup easier.

5. **Add a Remote Repository** (If you've just initialized a local repo or if
   it's not yet connected to a remote):

    ```bash
    ~/yolo $ git remote add origin <repo-url>
    ```

    Replace `<repo-url>` with your repository's URL, which you can obtain from
    GitHub or another Git service such as GitLab or Bitbucket.

6. **Securely Push to the Remote Using a Token** (Optional but recommended for
   enhanced security):

    Before pushing changes, especially when 2FA (Two Factor Authentication) is
    enabled, you might need to use a token instead of a password.

    ```bash
    ~/yolo $ git remote set-url origin https://<token>@github.com/<username>/<repository>
    ```

    Replace `<token>`, `<username>`, and `<repository>` with your personal
    access token, your GitHub username, and your repository name, respectively.

7. **Push Your Changes**:

    ```bash
    ~/yolo $ git push -u origin main
    ```

    This command pushes your commits to the `main` branch of the remote
    repository. The `-u` flag sets the upstream, making `origin main` the
    default target for future pushes.

#### Cloning an Existing Repository

If you've already cloned an existing repository, many of these steps
(specifically from initializing the repo to the first commit) are unnecessary
since the repository comes pre-initialized and connected to its remote
counterpart. You'd typically start by pulling the latest changes with `git pull`
and then proceed with your work.

#### Git Workflow

It is important to establish a consistent Git workflow to ensure that changes
are managed effectively and that the project's history is kept clean.

You can read more about git workflows here:

-   [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
-   [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
-   [Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)

### Virtual Environment

We can follow python's official documentation on
[installing packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
In what follows, we would give a brief overview of the steps to set up a virtual
environment for your project.

#### Create Virtual Environment

```bash
~/yolo $ python3 -m venv <venv-name>
```

#### Activate Virtual Environment

````{tab} Unix/macOS
```bash
~/yolo $ source <venv-name>/bin/activate
```
````

````{tab} Windows
```bash
~/yolo $ <venv-name>\Scripts\activate
```
````

#### Upgrade pip, setuptools and wheel

```bash
(venv) ~/yolo $ python3 -m pip install --upgrade pip setuptools wheel
```

### Managing Project Dependencies

Once you've established a virtual environment for your project, the next step is
to install the necessary libraries and packages. This process ensures that your
project has all the tools required for development and execution.

#### Managing Dependencies with `requirements.txt`

For project dependency management, the use of a `requirements.txt` file is a
common and straightforward approach. This file lists all the packages your
project needs, allowing for easy installation with a single command.

For simpler or moderately complex projects, a `requirements.txt` file is often
sufficient. Create this file and list each dependency on a separate line,
specifying exact versions to ensure consistency across different environments.

1. **Create a `requirements.txt` file**:

    ```bash
    touch requirements.txt
    ```

2. **Populate `requirements.txt` with your project's dependencies**. For
   example:

    ```plaintext
    torch==1.10.0+cu113
    torchaudio==0.10.0+cu113
    torchvision==0.11.1+cu113
    albumentations==1.1.0
    matplotlib==3.2.2
    pandas==1.3.1
    torchinfo==1.7.1
    tqdm==4.64.1
    wandb==0.12.6
    ```

    You can directly edit `requirements.txt` in your favorite text editor to
    include the above dependencies.

3. **Install the dependencies** from your `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

Certain libraries, like PyTorch with CUDA support, may require downloading
binaries from a specific URL due to additional dependencies. In such cases, you
can use the `-f` option with `pip` to specify a custom repository for dependency
links:

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

This command tells `pip` to also look at the given URL for any packages listed
in your `requirements.txt`, which is particularly useful for installing versions
of libraries that require CUDA for GPU acceleration.

You can also have a `requirements-dev.txt` file for development dependencies.

```bash
touch requirements-dev.txt
```

These dependencies are often used for testing, documentation, and other
development-related tasks.

#### Managing Dependencies with `pyproject.toml`

`pyproject.toml` is a configuration file introduced in
[PEP 518](https://www.python.org/dev/peps/pep-0518/) as a standardized way for
Python projects to manage project settings and dependencies. It aims to replace
the traditional `setup.py` and `requirements.txt` files with a single, unified
file that can handle a project's build system requirements, dependencies, and
other configurations in a standardized format.

-   **Unified Configuration**: `pyproject.toml` consolidates various tool
    configurations into one file, making project setups more straightforward and
    reducing the number of files at the project root.
-   **Dependency Management**: It can specify both direct project dependencies
    and development dependencies, similar to `requirements.txt` and
    `requirements-dev.txt`. Tools like `pip` can read `pyproject.toml` to
    install the necessary packages.
-   **Build System Requirements**: It explicitly declares the build system
    requirements, ensuring the correct tools are present before the build
    process begins. This is particularly important for projects that need to
    compile native extensions.
-   **Tool Configuration**: Many Python tools (e.g., `black`, `flake8`,
    `pytest`) now support reading configuration options from `pyproject.toml`,
    allowing developers to centralize their configurations.

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "example_project"
version = "0.1.0"
description = "An example project"
authors = [{name = "Your Name", email = "you@example.com"}]
dependencies = [
    "requests>=2.24",
    "numpy>=1.19"
]
optional-dependencies = {
    "dev" = ["pytest>=6.0", "black", "flake8"]
}

[tool.black]
line-length = 88
target-version = ['py38']

[tool.pytest]
minversion = "6.0"
addopts = "-ra -q"
```

Now if you want to install the dependencies, you can do so with:

```bash
pip install .
```

and for development dependencies:

```bash
pip install .[dev]
```

To this end, you should see the following directory structure:

```tree title="main directory tree" linenums="1" hl_lines="10 11 12 13 14 15"
.
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share
```

You can find more information on writing your
[`pyproject.toml` file here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## Phase 3. Build

1. Dockerize the application. Production identical environment.
2. Infrastructure as Code (IaC) for cloud deployment.
3. Container orchestration for scaling and managing containers.

## Styling and Formatting

## Pre-commit

This is on branch `pre-commit`.

```bash title="install pre-commit" linenums="1"
~/yolo (venv) $ pip install pre-commit
~/yolo (venv) $ pre-commit install

pre-commit installed at .git\hooks\pre-commit
```

```bash title="create pre-commit-config.yaml" linenums="1"
~/yolo (venv) $ touch .pre-commit-config.yaml
```

```yaml title=".pre-commit-config.yaml" linenums="1"
~/yolo (venv) $ cat .pre-commit-config.yaml

repos:
- repo: local
  hooks:
    - id: linter
      name: Run linter
      entry: bash
      args: ["./scripts/linter.sh"]
      language: system
      pass_filenames: false
    - id: check_format
      name: Run black code formatter
      entry: bash
      args: ["./scripts/formatter.sh"]
      language: system
      pass_filenames: false
```

means we will run `linter.sh` and `formatter.sh` before every commit.

### Folder Structure

A very barebone structure as of now would be as follows:

```tree title="pipeline-feature" linenums="1"
.
├── LICENSE
├── README.md
├── mlops_pipeline_feature_v1
│   ├── __init__.py
│   └── pipeline.py
├── pyproject.toml
├── requirements.txt
└── requirements_dev.txt
```

## Documentation

### Jupyter Book Setup

```bash title="packages required for jupyter-book" linenums="1"
jupyter-book==0.13.1
sphinx-inline-tabs==2021.3.28b7
sphinx-proof==0.1.3
myst-nb==0.16.0 # remember to download manually
```

```bash title="create jupyter-book" linenums="1"
~/yolo (venv) $ mkdir content
```

You populate the `content` folder with your notebooks and markdown files.

To build the book, run:

```bash title="build jupyter-book" linenums="1"
~/yolo (venv) $ jupyter-book build content
```

Then the book will be built in the `_build` folder.

Lastly, to serve and deploy the book, run:

```bash title="serve and deploy jupyter-book" linenums="1"
~/yolo (venv) $ mkdir .github/workflows
~/yolo (venv) $ touch .github/workflows/deploy.yml
```

```bash title="deploy.yml" linenums="1"
~/yolo (venv) $ cat .github/workflows/deploy.yml

name: deploy

on:
  # Trigger the workflow on push to main branch
  push:
    branches:
      - main
      - master

# This job installs dependencies, build the book, and pushes it to `gh-pages`
jobs:
  build-and-deploy-book:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: [3.8]
    steps:
    - uses: actions/checkout@v2

    # Install dependencies
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v1
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    # Build the book
    - name: Build the book
      run: |
        jupyter-book build content

    # Deploy the book's HTML to gh-pages branch
    - name: GitHub Pages action
      uses: peaceiris/actions-gh-pages@v3.6.1
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: content/_build/html
```

This will deploy the book to the `gh-pages` branch. Remember to enable GitHub
Pages in the repository settings.

### Mkdocs Setup

We will be using [Mkdocs](https://www.mkdocs.org/) to generate our markdown
documentation into a static website.

1. The following requirements are necessary to run `mkdocs`:

    ```txt title="requirements.txt" linenums="1"
    mkdocs                            1.3.0
    mkdocs-material                   8.2.13
    mkdocs-material-extensions        1.0.3
    mkdocstrings                      0.18.1
    ```

2. Initialize default template by calling `mkdocs new .` where `.` refers to the
   current directory. The `.` can be replaced with a path to your directory as
   well. Subsequently, a folder `docs` alongside with `mkdocs.yml` file will be
   created.

    ```tree title="mkdocs folder structure" linenums="1" hl_lines="3 4 5"
    pkd_exercise_counter/
    ├── venv_pkd_exercise_counter/
    ├── docs/
    │   └── index.md
    ├── mkdocs.yml
    ├── requirements.txt
    └── setup.py
    ```

3. We can specify the following configurations in `mkdocs.yml`:

    ???+ example "Show/Hide mkdocs.yml"
    `yml title="mkdocs.yml" linenums="1" site_name: Hongnan G. PeekingDuck Exercise Counter site_url: "" nav: - Home: index.md - PeekingDuck: - Setup: workflows.md - Push-up Counter: pushup.md theme: name: material features: - content.code.annotate markdown_extensions: - attr_list - md_in_html - admonition - footnotes - pymdownx.highlight - pymdownx.inlinehilite - pymdownx.superfences - pymdownx.snippets - pymdownx.details - pymdownx.arithmatex: generic: true extra_javascript: - javascript/mathjax.js - https://polyfill.io/v3/polyfill.min.js?features=es6 - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js extra_css: - css/extra.css plugins: - search - mkdocstrings # plugins for mkdocstrings `

    Some of the key features include:

    - [Code block Line Numbering](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/);
    - [Code block Annotations](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/);
    - [MathJax](https://squidfunk.github.io/mkdocs-material/reference/mathjax/).

    One missing feature is the ability to **toggle** code blocks. Two
    workarounds are provided:

    ??? "Toggle Using Admonition"
    `bash title="Setting Up" mkdir custom_hn_push_up_counter `

    <details>
    <summary>Toggle Using HTML</summary>
    ```bash title="Setting Up"
    mkdir custom_hn_push_up_counter
    ```
    </details>

4. We added some custom CSS and JavaScript files. In particular, we added
   `mathjax.js` for easier latex integration.
5. You can now call `mkdocs serve` to start the server at a local host to view
   your document.

!!! tip To link to a section or header, you can do this: [link to Styling and
Formatting by
[workflow.md#styling-and-formatting](workflow.md#styling-and-formatting).

### Mkdocstrings

We also can create docstrings as API reference using
[Mkdocstrings](https://mkdocstrings.github.io/usage/):

-   Install mkdocstrings: `pip install mkdocstrings`
-   Place plugings to `mkdocs.yml`:
    ```yml title="mkdocs.yml" linenums="1"
    plugins:
        - search
        - mkdocstrings
    ```
-   In `mkdocs.yml`'s navigation tree:

    ```yml title="mkdocs.yml" linenums="1"
    - API Documentation:
          - Exercise Counter: api/exercise_counter_api.md
    ```

    For example you have a python file called `exercise_counter.py` and want to
    render it, create a file named `api/exercise_counter_api.md` and in this
    markdown file:

    ```md title="api/exercise_counter_api.md" linenums="1"
    ::: custom_hn_exercise_counter.src.custom_nodes.dabble.exercise_counter #
    package path.
    ```

## Tests

Set up `pytest` for testing codes.

```bash title="Install pytest" linenums="1"
pytest==6.0.2
pytest-cov==2.10.1
```

In general, **Pytest** expects our testing codes to be grouped under a folder
called `tests`. We can configure in our `pyproject.toml` file to override this
if we wish to ask `pytest` to check from a different directory. After specifying
the folder holding the test codes, `pytest` will then look for python scripts
starting with `tests_*.py`; we can also change the extensions accordingly if you
want `pytest` to look for other kinds of files
(extensions)[^testing_made_with_ml].

```bash title="pyproject.toml" linenums="1"
# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

[^testing_made_with_ml]:
    This part is extracted from
    [madewithml](https://madewithml.com/courses/mlops/testing/#pytest).

## CI/CD (GitHub Actions)

The following content is with reference to:

-   [MLOps Basics [Week 6]: CI/CD - GitHub Actions](https://www.ravirajag.dev/blog/mlops-github-actions)
-   [CI/CD for Machine Learning](https://madewithml.com/courses/mlops/cicd/)

We will be using [GitHub Actions](https://github.com/features/actions) to setup
our mini CI/CD.

### Pre-Merge Checks

Commit checks is to ensure the following:

-   The requirements can be installed on various OS and python versions.
-   Ensure code quality and adherence to PEP8 (or other coding standards).
-   Ensure tests are passed.

```yaml title="lint_test.yml" linenums="1"
name: Commit Checks # (1)
on: [push, pull_request] # (2)

jobs: # (3)
    check_code: # (4)
        runs-on: ${{ matrix.os }} # (5)
        strategy: # (6)
            fail-fast: false # (7)
            matrix: # (8)
                os: [ubuntu-latest, windows-latest] # (9)
                python-version: [3.8, 3.9] # (10)
        steps: # (11)
            - name: Checkout code # (12)
              uses: actions/checkout@v2 # (13)
            - name: Setup Python # (14)
              uses: actions/setup-python@v2 # (15)
              with: # (16)
                  python-version: ${{ matrix.python-version }} # (17)
                  cache: "pip" # (18)
            - name: Install dependencies # (19)
              run: | # (20)
                  python -m pip install --upgrade pip setuptools wheel
                  pip install -e .
            - name: Run Black Formatter # (21)
              run: black --check . # (22)
            # - name: Run flake8 Linter
            #   run: flake8 . # look at my pyproject.toml file and see if there is a flake8 section, if so, run flake8 on the files in the flake8 section
            - name: Run Pytest # (23)
              run: python -m coverage run --source=custom_hn_exercise_counter -m
                  pytest && python -m coverage report # (24)
```

1.  This is the name that will show up under the **Actions** tab in GitHub.
    Typically, we should name it appropriately like how we indicate the subject
    of an email.
2.  The list here indicates the
    [workflow will be triggered](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request)
    whenever someone directly pushes or submits a PR to the main branch.
3.  Once an event is triggered, a set of **jobs** will run on a
    [runner](https://github.com/actions/runner). In our example, we will run a
    job called `check_code` on a runner to check for formatting and linting
    errors as well as run the `pytest` tests.
4.  This is the name of the job that will run on the runner.
5.  We specify which OS system we want the code to be run on. We can simply say
    `ubuntu-latest` or `windows-latest` if we just want the code to be tested on
    a single OS. However, here we want to check if it works on both Ubuntu and
    Windows, and hence we define `${{ matrix.os }}` where `matrix.os` is
    `[ubuntu-latest, windows-latest]`. A cartesian product is created for us and
    the job will run on both OSs.
6.  Strategy is a way to control how the jobs are run. In our example, we want
    the job to run as fast as possible, so we set `strategy.fail-fast` to
    `false`.
7.  If one job fails, then the whole workflow will fail, this is not ideal if we
    want to test multiple jobs, we can set `fail-fast` to `false` to allow the
    workflow to continue running on the remaining jobs.
8.  Matrix is a way to control how the jobs are run. In our example, we want to
    run the job on both Python 3.8 and 3.9, so we set `matrix.python-version` to
    `[3.8, 3.9]`.
9.  This list consists of the OS that the job will run on in cartesian product.
10. This is the python version that the job will run on in cartesian product. We
    can simply say `3.8` or `3.9` if we just want the code to be tested on a
    single python version. However, here we want to check if it works on both
    python 3.8 and python 3.9, and hence we define
    `${{ matrix.python-version }}` where `matrix.python-version` is
    `[3.8, 3.9]`. A cartesian product is created for us and the job will run on
    both python versions.
11. This is a list of dictionaries that defines the steps that will be run.
12. Name is the name of the step that will be run.
13. It is important to specify `@v2` as if unspecified, then the workflow will
    use the latest version from actions/checkout template, potentially causing
    libraries to break. The idea here is like your `requirements.txt` idea, if
    different versions then will break.
14. Setup Python is a step that will be run before the job.
15. Same as above, we specify `@v2` as if unspecified, then the workflow will
    use the latest version from actions/setup-python template, potentially
    causing libraries to break.
16. With is a way to pass parameters to the step.
17. This is the python version that the job will run on in cartesian product and
    if run 1 python version then can define as just say 3.7
18. Cache is a way to control how the libraries are installed.
19. Install dependencies is a step that will be run before the job.
20. `|` is multi-line string that runs the below code, which sets up the
    libraries from `setup.py` file.
21. Run Black Formatter is a step that will be run before the job.
22. Runs `black` with configurations from `pyproject.toml` file.
23. Run Pytest is a step that will be run before the job.
24. Runs pytest, note that I specified `python -m` to resolve PATH issues.

### Deploy to Website

The other workflow for this project is to deploy the website built from Mkdocsto
gh-pages branch.

??? example "Show/Hide content for deploy_website.yml" ```yaml
title="deploy_website.yml" linenums="1" name: Deploy Website to GitHub Pages

    on:
      push:
        branches: [master]
        paths:
          - "docs/**"
          - "mkdocs.yml"
          - ".github/workflows/deploy_website.yml"

    permissions: write-all

    jobs:
      deploy:
        runs-on: ubuntu-latest
        name: Deploy Website
        steps:
          - uses: actions/checkout@v2
          - name: Set Up Python
            uses: actions/setup-python@v2
            with:
              python-version: 3.8
              architecture: x64
          - name: Install dependencies
            run: | # this symbol is called a multiline string
              python -m pip install --upgrade pip setuptools wheel
              pip install -e .

          - name: Build Website
            run: |
              mkdocs build
          - name: Push Built Website to gh-pages Branch
            run: |
              git config --global user.name 'Hongnan G.'
              git config --global user.email 'reighns92@users.noreply.github.com'
              ghp-import \
              --no-jekyll \
              --force \
              --no-history \
              --push \
              --message "Deploying ${{ github.sha }}" \
              site
    ```

## Phase 5. Continuous Deployment

### Release

## Phase 6. Continuous Monitoring

### Monitoring and Observability

### Motivation

-   We may not be able to catch all the bugs and errors in the code.
-   Failure with trace.
