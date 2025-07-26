---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 10. Continuous Integration, Deployment, Learning and Training (DevOps, DataOps, MLOps)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)

```{contents}
```

```{figure} ./assets/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg
---
name: mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd-stage-10
---

CI/CD and automated ML pipeline.

Image Credits: [Google - MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
```

## Continuous Integration (CI)

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

## Technical Debt is Beyond Bad Code

First of all, one should move away from the **_prior_** that _technical debt_ is
_equivalent_ to _bad code_[^stop_saying_technical_debt]. While this may be often
true, **equating** technical debt to bad code is a _simplification_ that, if not
addressed, can lead to _misunderstanding_ and _miscommunication_. If one shows
up and spurs out the contrapositive of the above statement, to derive to the
conclusion that _if one writes good code, then there is no technical debt_, and
in turn, there's no need to **revisit**, **document**, **lint**, and **test**
the code[^stop_saying_technical_debt], and even though logically valid, he will
be in for a _rude awakening_.

Notwithstanding the slightly _philosophical_ introduction, the _practical
situations_ are in fact, no one writes perfect code, and even just having a
small amount of "bad code" will accumulate, some may not be even _immediately
obvious_. If we do not have an _assurance policy_ (read: **Continuous
Integration/Continuous Deployment**) in place, then what we end up is a
_codebase_ full of _technical debt_, and trust me, no developer wants to inherit
such a _codebase_.

On a more _serious_ note, the **broader implication** of _technical debt_ is
that the application sitting on top of it is _unreliable_, _unstable_, and
_insecure_, leading to _bugs_, _security vulnerabilities_, and _performance
issues_. What's worse is if one ships this code to _production_, then the _cost_
of _fixing_ the _bugs_ is _exponentially higher_ than if it was _fixed_ in the
_development_ phase. Fortunately, any _competent tech organization_ will have a
_CI/CD_ pipeline in place, and will go through various stages of _testing_ and
_validation_ before shipping to _production_.

Before we move on, we need to be **_crystal clear_** that **CI/CD** is not a
**silver bullet** that will **_eradicate_** all _bugs_ and _technical debt_.
After all, we often see code breaking in _production_, and the _reason_ for this
is two folds:

-   How you implement your CI/CD pipeline matters. Skipping certain stages such
    as _type safety checks_ will inevitably lead to _bugs_ in dynamic languages
    like _Python_.
-   How you write your code matters - consider the case where your unit tests
    are **_not robust_**, then it obviously will not catch the bugs that it is
    supposed to catch.

Even if you think you fulfill the above two, we quote the well known saying
**_To err is human_**, to play at the fact that _bugs_ are _inevitable_, and
that means there is no such thing as a 100% bug free codebase. The argument here
is not to **_eradicate_** bugs, but to **_reduce_** them - _a numbers game_.

## Lifecycle

Insert [image]?

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

#### README

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

#### LICENSE

The `LICENSE` file is critical as it defines how others can legally use, modify,
and distribute your project. If you’re unsure which license to use,
[choosealicense.com](https://choosealicense.com/) can help you decide.

```bash
~/yolo $ touch LICENSE
```

After creating the file, you should fill it with the text of the license you've
chosen. This could be the MIT License, GNU General Public License (GPL), Apache
License 2.0, etc.

#### CONTRIBUTING

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

[**Version control**](https://en.wikipedia.org/wiki/Version_control), such as
[**Git**](https://git-scm.com/), are foundational to modern software development
practices, including Continuous Integration and Continuous Deployment/Delivery
(CI/CD). The premise to CI/CD is to enable developers to work on the same code
base simultaneously, track every change, and automate the CI/CD processes such
as triggering builds, tests, and deployments based on code commits and merges.

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

    ```text
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

    ```text
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

```text title="main directory tree" linenums="1" hl_lines="10 11 12 13 14 15"
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

#### Pinning DevOps Tool Versions

In DevOps, particularly in continuous integration (CI) environments, pinning
exact versions of tools like `pytest`, `mypy`, and other linting tools is
important. Here are the key reasons:

1. **Reproducibility**: Pinning specific versions ensures that the development,
   testing, and production environments are consistent. This means that code
   will be tested against the same set of dependencies it was developed with,
   reducing the "it works on my machine" problem.

2. **Stability**: Updates in these tools can introduce changes in their behavior
   or new rules that might break the build process. By pinning versions, you
   control when to upgrade and prepare for any necessary changes in your
   codebase, rather than being forced to deal with unexpected issues from
   automatic updates.

Tools like `black`, `isort`, `mypy`, and `pylint` are particularly important to
pin because they directly affect code quality and consistency. Changes in their
behavior due to updates can lead to new linting errors or formatting changes
that could disrupt development workflows.

```{prf:example} Pinning Pylint Version
:label: cicd-concept-pinning-pylint

Consider the Python linting tool`pylint`. It's known for its thoroughness in
checking code quality and conformity to coding standards. However, `pylint` is
also frequently updated, with new releases potentially introducing new checks or
modifying existing ones.

Suppose your project is using `pylint` version 2.6.0. In this version, your
codebase passes all linting checks, ensuring a certain level of code quality and
consistency. Now, imagine `pylint` releases a new version, 2.7.0, which includes
a new check for a particular coding pattern (e.g., enforcing more stringent
rules on string formatting or variable naming).

Consequently, if you _don't_ pin the `pylint` version, the next time you run
an installation (e.g., `pip install -r requirements.txt` in local or remote),
`pylint` has a good chance of being updated to version 2.7.0. This update could
trigger new linting errors in your codebase, even though the code itself hasn't
changed. This situation can be particularly disruptive in a CI/CD environment,
where a previously passing build now fails due to new linting errors.
```

### Local Pre-Commit Checks (Local Continuous Integration)

Consider continuous integration (CI) as a practice of merging all developers'
working copies to a shared mainline several times a day. Each integration is
verified by an automated build (including linting, code smell, type checks, unit
tests etc) to detect integration errors as quickly as possible. This build can
be triggered easily by modern version control systems like Git through a simple
push to the repository. Tools like GitHub Actions, a CI/CD feature within
GitHub, play the role in facilitating these practices.

As we will continuously emphasize, to maximize the effectiveness of tools like
Jenkins and GitHub Actions, it's crucial to maintain **consistency** and
**uniformity** between the local development environment and the CI/CD pipeline.
This alignment ensures that the software builds, tests, and deploys in an
identical manner, both locally and in the CI/CD environment. Achieving this
uniformity often involves the use of containerization tools like Docker, which
can encapsulate the application and its dependencies in a container that runs
consistently across different computing environments. By doing so, developers
can minimize the 'it works on my machine' syndrome, a common challenge in
software development, and foster a more collaborative and productive development
culture. Moreover, it is also common to see developers get a "shock" when their
build failed in the remote CI/CD pipeline, which could have been easily detected
locally _**if and only if**_ they had run the **same** checks locally. There are
some commercial tools that may not be able to same checks locally, but for
open-source tools, it is a good practice to run the same checks locally.

Some developers will "forget" to run the same checks locally, and this is where
[**pre-commit hooks**](https://pre-commit.com/) come into play. Pre-commit hooks
are scripts that run before a commit is made to the version control system.

#### Guard Rails

As the name suggests, guard rails are a set of rules and guidelines that help
developers stay on track and avoid common security, performance, and
maintainability pitfalls. These guard rails can be implemented as pre-commit
hooks, which are scripts that run before a commit is made to the version control
system.

-   [Bandit](https://bandit.readthedocs.io/en/latest/config.html) is a tool
    designed to find common security issues in Python code. To install Bandit,
    run:

    ```bash
    pip install -U bandit
    ```

    and you can place configurations of Bandit in a `.bandit` file. But more
    commonly, we put in `pyproject.toml` file for unification.

    ```toml
    # FILE: pyproject.toml
    [tool.bandit]
    exclude_dirs = ["tests", "path/to/file"]
    tests = ["B201", "B301"]
    skips = ["B101", "B601"]
    ```

-   [`detect-secrets`](https://github.com/Yelp/detect-secrets/tree/master) is a
    tool that can be used to prevent secrets from being committed to your
    repository. It can be installed using pip:

    ```bash
    pip install -U detect-secrets
    ```

    You can find more information in the
    [usage section](https://github.com/Yelp/detect-secrets/tree/master). People
    commonly place this as a hook in the
    [`.pre-commit-config.yaml`](https://github.com/Yelp/detect-secrets/blob/master/.pre-commit-hooks.yaml)
    file.

-   [Safety](https://github.com/pyupio/safety) is a tool that checks your
    dependencies for known security vulnerabilities. You can draw parallels to
    Nexus IQ, which is a commercial tool that does the same thing.

There are many more guard rails that can be implemented as pre-commit hooks, we
cite
[Welcome to pre-commit heaven - Marvelous MLOps Substack](https://marvelousmlops.substack.com/i/130911126/guard-rails)
as a good reference below:

```yaml
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.3.0
      hooks:
          - id: check-ast
          - id: check-added-large-files
          - id: check-json
          - id: check-toml
          - id: check-yaml
          - id: check-shebang-scripts-are-executable
          - id: detect-secrets
    - repo: https://github.com/PyCQA/bandit
      rev: 1.7.4
      hooks:
          - id: bandit
```

#### Styling, Formatting, and Linting

Guido Van Rossum, the author of Python, aptly stated, "Code is read more often
than it is written." This principle underscores the necessity of both clear
documentation and easy readability in coding. Adherence to style and formatting
conventions, particularly those based on
[PEP8](https://peps.python.org/pep-0008/), plays a vital role in achieving this
goal. Different teams may adopt various conventions, but the key lies in
consistent application and the use of automated pipelines to maintain this
consistency. For instance, standardizing line lengths simplifies code review
processes, making discussions about specific sections more straightforward. In
this context, **linting** and **formating** emerge as critical tools for
maintaining high code quality. Linting, the process of analyzing code for
potential errors, and formatting, which ensures a uniform appearance,
collectively boost **readability** and **maintainability**. A well-styled
codebase not only looks professional but also reduces bugs and eases
**integration** and **code reviews**. These practices, when ingrained as an
**intuition** among developers, lead to more robust and efficient software
development.

This part is probably what most people are familiar with, we list some common
tools for styling, formatting, and linting below (cited from
[Welcome to pre-commit heaven - Marvelous MLOps Substack](https://marvelousmlops.substack.com/i/130911126)):

```yaml
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.3.0
      hooks:
          - id: end-of-file-fixer
          - id: mixed-line-ending
          - id: trailing-whitespace
    - repo: https://github.com/psf/black
      rev: 22.10.0
      hooks:
          - id: black
            language_version: python3.11
            args:
                - --line-length=128
          - id: black-jupyter
            language_version: python3.11
    - repo: https://github.com/pycqa/isort
      rev: 5.11.5
      hooks:
          - id: isort
            args: ["--profile", "black"]
    - repo: https://github.com/pycqa/flake8
      rev: 5.0.4
      hooks:
          - id: flake8
            args:
                - "--max-line-length=128"
            additional_dependencies:
                - flake8-bugbear
                - flake8-comprehensions
                - flake8-simplify
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v0.991
      hooks:
          - id: mypy
```

#### Tests

Testing is a critical part of the software development process. It helps ensure
that the code behaves as expected and that changes don't introduce new bugs or
break existing functionality. Unit tests, in particular, focus on testing
individual components or units of code in isolation. They help catch bugs early
and provide a safety net for refactoring and making changes with confidence.

Usually, unit tests are run as part of _pre-merge checks_ to ensure that the
changes being merged don't break existing functionality where as _post-merge
checks_ can entail more comprehensive tests such as integration tests,
end-to-end tests etc.

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

```{admonition} References
:class: seealso

- [Chapter 5. Testing - Py-Pkgs](https://py-pkgs.org/05-testing)
```

#### Git Sanity Checks

Git sanity checks are a set of rules and guidelines that help developers avoid
common mistakes and pitfalls when working with Git. More specifically, we have
the below:

-   **commitizen**: This hook encourages developers to use the Commitizen tool
    for formatting commit messages. Commitizen standardizes commit messages
    based on predefined conventions, making the project's commit history more
    readable and navigable. Standardized messages facilitate understanding the
    purpose of each change, aiding in debugging and project management (though I
    rarely need to sieve through commit messages) but this is good practice
    (imagine all your commit message is "111" or "fix bug" or "update").

-   **commitizen-branch**: A specific use of the Commitizen validation that can
    be configured to work at different stages, such as during branch pushes.
    This ensures that commits pushed to branches also follow the standardized
    format, maintaining consistency not just locally but across the repository.

-   **check-merge-conflict**: This hook checks for merge conflict markers (e.g.,
    `<<<<<<<`, `=======`, `>>>>>>>`). These markers indicate unresolved merge
    conflicts, which should not be committed to the repository as they can break
    the codebase. Preventing such commits helps maintain the integrity and
    operability of the project.

-   **no-commit-to-branch**: It prevents direct commits to specific branches
    (commonly the main or master branch). This practice encourages the use of
    feature branches and pull requests, fostering code reviews and discussions
    before changes are merged into the main codebase. It's a way to ensure that
    changes are vetted and tested, reducing the risk of disruptions in the main
    development line.

Again, citing from
[Welcome to pre-commit heaven - Marvelous MLOps Substack](https://marvelousmlops.substack.com/i/130911126):

```yaml
repos:
  - repo: https://github.com/commitizen-tools/commitizen
      rev: v2.35.0
      hooks:
        - id: commitizen
        - id: commitizen-branch
          stages: [push]
  - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.3.0
      hooks:
        - id: check-merge-conflict
        - id: no-commit-to-branch
```

#### Code Correctors

This is entering "riskier" territory, as code correctors can automatically
correct your code.

-   `pyupgrade` is a tool that automatically upgrades Python syntax to the
    latest version that's supported by the Python interpreter specified. It
    takes existing Python code and refactors it where possible to use newer
    syntax features that are more efficient, readable, or otherwise preferred.
    For instance, when targeting Python 3.9 and above, it might convert
    old-style string formatting to f-strings, use newer Python 3.9 dictionary
    merge operators, and more.

-   `yesqa` automatically removes unnecessary `# noqa` comments from the code.
    `# noqa` is used to tell linters to ignore specific lines of code that would
    otherwise raise warnings or errors. However, over time, as the code evolves,
    some of these `# noqa` comments might no longer be necessary because the
    issues have been resolved or the code has changed.

#### Setting Up Pre-Commit

1.  **Install Pre-Commit**:

    ```bash title="install pre-commit" linenums="1"
    ~/yolo (venv) $ pip install -U pre-commit
    ~/yolo (venv) $ pre-commit install
    ```

2.  **Create a `.pre-commit-config.yaml` File**:

    ```bash title="create pre-commit-config.yaml" linenums="1"
    ~/yolo (venv) $ touch .pre-commit-config.yaml
    ```

3.  **Populate `.pre-commit-config.yaml` with the desired hooks**:

    Sample `.pre-commit-config.yaml` file:

    ```yaml title=".pre-commit-config.yaml" linenums="1"
    repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.5.0
        hooks:
        - id: check-added-large-files
        - id: check-ast
        - id: check-builtin-literals
        - id: check-case-conflict
        - id: check-docstring-first
        - id: check-executables-have-shebangs
        - id: check-json
        - id: check-shebang-scripts-are-executable
        - id: check-symlinks
        - id: check-toml
        - id: check-vcs-permalinks
        - id: check-xml
        - id: check-yaml
        - id: debug-statements
        - id: destroyed-symlinks
        - id: mixed-line-ending
        - id: trailing-whitespace
    ```

4.  **Run Pre-Commit**:

    Sample command to run pre-commit on all files:

    ```bash title="run pre-commit" linenums="1"
    ~/yolo (venv) $ pre-commit run --all-files
    ```

    This command runs all the hooks specified in the `.pre-commit-config.yaml`
    file on all files in the repository.

### Documentation

Documentation is severely underlooked in many organizations. However,
documentation is a fundamental part of the development process as it provides
guidance for users and developers through carefully crafted explanations,
cookbooks, tutorials and API references.

The documentation should necessarily be part of the CI/CD pipeline, and
everytime you update the documentation, it should be automatically built and
deployed to the documentation hosting platform.

-   [Sphinx](https://www.sphinx-doc.org/en/master/): A tool that makes it easy
    to create intelligent and beautiful documentation for Python projects. It is
    commonly used to document Python libraries and applications, but it can also
    be used to document other types of projects.
-   [MkDocs](https://www.mkdocs.org/): A fast, simple, and downright gorgeous
    static site generator that's geared towards building project documentation.
-   [Sphinx API Documentation](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html):
    A tool that automatically generates API documentation from your source code.
    It's commonly used in conjunction with Sphinx to create API references for
    Python projects.

```{admonition} References
:class: seealso

-   [Chapter 6. Documentation - Py-Pkgs](https://py-pkgs.org/06-documentation)
-   [Hypermodern Python Chapter 5: Documentation - Claudio Jolowicz](https://cjolowicz.github.io/posts/hypermodern-python-05-documentation/)
```

### A Word on Type Safety Checks in Python

#### Python is Strongly and Dynamically Typed

First of all, one should be clear that Python is considered a _strongly_ and
_dynamically_ typed language[^python_strongly_and_dynamic_typing].

_Dynamic_ because runtime objects have a type, as opposed to the variable having
a type in statically typed languages. For example, consider a variable
`str_or_int` in python:

```python
str_or_int: str | int = "hello"
```

and later in the code, we can reassign it to an integer:

```python
str_or_int: str | int = 42
```

This is not possible in statically typed languages, where the type of a variable
is fixed at compile time.

_Strongly_ because Python does not allow unreasonable arbitrary type
conversions/coercions. You are not allowed to concatenate a string with an
integer, for example:

```python
"hello" + 42
```

will raise a `TypeError` exception.

#### MyPy: Static Type Checking

Static type checking is the process of verifying the type safety of a program
based on analysis of some source code. `mypy` is probably the most recognized
static type checker, in which it analyzes your code without executing it,
identifying type mismatches based on the type hints you've provided. mypy is
most valuable during the development phase, helping catch type-related errors
before runtime.

Consider the following very simply example:

```python
def concatenate(a: str, b: str) -> str:
    return a + b
```

Here, we've used type hints to specify that `a` and `b` are strings, and that
the function returns a string. A subtle bug can be introduced just for the sake
of illustration:

```python
concatenate("hello", 42) # mypy will catch this
```

This will raise a `TypeError` exception at runtime, but mypy will catch this
before the code is executed. One would argue that this is a trivial example,
since runtime will catch this error. However, in a larger codebase, you may not
have the luxury of running the entire codebase to catch such errors.
Furthermore, there are even more _silent_ errors that runtime may not catch due
to a certain combination that does not immediately raise an exception.

#### TypeGuard: Runtime Type Checking

Unlike `mypy`, `typeguard` operates at runtime. In other words, `mypy` just tell
you "this is wrong" but `typeguard` will actually raise an error at runtime.
Consequently, we can view `typeguard` as a runtime type checker and
complementary to `mypy`.

`typeguard` is particularly useful in testing scenarios or in development
environments where you want to ensure that type annotations are being respected
at runtime. It can catch type errors that static analysis might not catch due to
the dynamic nature of Python or when dealing with external systems that might
not adhere to the expected types.

To use TypeGuard, you typically employ it through decorators or with type
checking in unit tests.

```python
from typeguard import typechecked

@typechecked
def greet(name: str) -> str:
    return 'Hello ' + name

try:
    greet(42)
except TypeError as e:
    print(e)
```

### Release and Versioning

```{admonition} References
:class: seealso

-   [Chapter 7. Releasing and versioning - Py-Pkgs](https://py-pkgs.org/07-releasing-versioning)
```

## Phase 3. Build

Considering we have came so far in the development phase, where we have set up
our main directory, version control, virtual environment, project dependencies,
local pre-commit checks, documentation, and release and versioning, we can now
touch on the build phase - where _**automation**_ is the key. We need a way to
automate whatever we have done in the development phase, and feedback whether
the local build is actually "really" successful or not to every team member.

1. Dockerize the application. Production identical environment.
2. Infrastructure as Code (IaC) for cloud deployment.
3. Container orchestration for scaling and managing containers.

### Pre-Merge Checks

Commit checks is to ensure the following:

-   The requirements can be installed on various OS and python versions.
-   Ensure code quality and adherence to PEP8 (or other coding standards).
-   Ensure tests are passed.

```yaml title="lint_test.yaml" linenums="1"
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

### Orchestration

Here you would have many github actions/jenkins workflows that orchestrate the
build, test, and release of your application. You might also have another
workflow to deploy your data pipeline to say a container registry like AWS ECR
or Google GCR.

### Infrastructure as Code (IaC)

You can also define templates for your infrastructure as code (IaC) in
[Terraform](https://www.terraform.io/docs/language/index.html).

## Phase 4. Scan and Test

First and foremost, this piece of article is not to teach you **_how_** to write
tests, because that would take a whole book to cover. Writing tests is "easy",
but writing **_good tests_** is an art, and difficult to master. I want to set
the stage for you to understand the importance of testing, and what types of
testing are there, along with some intuition.

### The Testing Pyramid

The testing pyramid is a visual metaphor that illustrates the ideal distribution
of testing methodologies from the base up: starting with unit tests, followed by
integration tests, then system testing, and capped off with end-to-end (E2E)
tests. This structure emphasizes the importance of a bottom-up approach to
testing, where the majority of tests are low-level, quick, and automated unit
tests, progressing to fewer, more comprehensive, and often manual tests at the
top.

-   Unit Tests
-   Integration Tests
-   System Tests
-   End-to-End Tests

```{figure} ./assets/pyramid-progression.jpg
---
name: devops-continuous-integration-testing-pyramid
---

The Testing Pyramid

**Image Credit:**
[Testing Pyramid](https://semaphoreci.com/blog/testing-pyramid)
```

### Unit Testing

Unit testing is a fundamental tool in every developer's toolbox. Unit tests not
only help us test our code, they encourage good design practices, reduce the
chances of bugs reaching production, and can even serve as examples or
documentation on how code functions. Properly written unit tests can also
improve developer efficiency.

#### Intuition

Unit tests are the smallest and most granular tests in the testing pyramid. This
can be explained through an analogy of a building. If you think of your
application as a building, unit tests are the bricks. They are the smallest,
most fundamental building blocks of your application. They test the smallest
pieces of code, such as functions, methods, or classes, in isolation from the
rest of the application.

You need to ensure each brick is solid and reliable before you can build a
sturdy, reliable building. Similarly, you need to ensure each unit of code is
solid and reliable before you can build a sturdy, reliable application.

#### Benefits of Unit Testing

##### Early Bug Detection and Reduce Cost

Why can't I catch a bug when the application is in production? Cost. It is
expensive to revert back and fix the bug. It is much cheaper to fix the bug when
it is caught early in the development cycle. Unit tests allow for the detection
of problems early in the development cycle, saving time and effort by preventing
bugs from propagating to later stages.

A
[2008 research study by IBM](https://www.researchgate.net/figure/IBM-System-Science-Institute-Relative-Cost-of-Fixing-Defects_fig1_255965523)
estimates that a bug caught in production could cost 6 times as much as if it
was caught during implementation[^unit-test-1].

##### Refactoring with Confidence

Development is an **_iterative process_**. You write code, test it, and then
refactor it. You repeat this process until you are satisfied with the result.

With a suite of unit tests, developers can make changes to the codebase
confidently, knowing that they'll be alerted if a change inadvertently breaks
something that used to work.

##### Unit Test As Documentation

Unit tests serve as a form of documentation that describes what the code is
supposed to do, helping new developers understand the project's functionality
more quickly.

#### Dependency Injection

Dependency Injection (DI) is a design pattern used to manage dependencies
between objects in software development. It's a technique that allows a class's
dependencies to be injected into it from the outside rather than being hardcoded
within the class. This approach promotes loose coupling, enhances testability,
and improves code maintainability.

At its core, DI involves three key components:

1. **The Client**: The object that depends on the service(s).
2. **The Injector**: The mechanism that injects the service(s) into the client.
3. **The Service**: The dependency or service being used by the client.

Dependency Injection can be implemented in several ways, including constructor
injection, setter injection, and interface injection. Each method has its
context and use case, but they all serve the same purpose: to decouple the
creation of a dependency from its usage.

##### Link to Unit Testing

The link between Dependency Injection and unit testing is fundamentally about
making code easier to test. DI facilitates the testing process in several ways:

-   **Isolation of Unit Tests**: By injecting dependencies into a class, you can
    easily replace those dependencies with mocks or stubs during testing. This
    allows you to isolate the unit of code being tested, ensuring that tests are
    not affected by external factors such as databases, file systems, or network
    calls.
-   **Flexibility in Test Scenarios**: Dependency Injection makes it easier to
    create different configurations of an object for testing. You can inject
    different implementations of a dependency to test how the object behaves
    under various conditions, enhancing test coverage and robustness.
-   **Reduced Boilerplate Code**: Without DI, you might find yourself writing a
    lot of boilerplate code to set up objects for testing, especially if they
    have numerous and complex dependencies. DI frameworks can automate much of
    this setup, keeping your test code cleaner and focused on the behavior
    you're testing.

##### No Dependency Injection vs Dependency Injection

Consider a simple class that processes user data and requires a database
connection to store this data. Without DI, the class might directly instantiate
a connection to a specific database, making it difficult to test the class
without accessing the actual database.

```{code-cell} ipython3
from typing import Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class DatasetLoader(Protocol):
    def load_feature_and_label(self) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        ...


class ImageDatasetLoader:
    def load_feature_and_label(self) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        features = np.array([[0.1, 0.2, 0.3]])
        labels = np.array([1])

        return features, labels


class TextDatasetLoader:
    def load_feature_and_label(self) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        features = np.array([[0.4, 0.5, 0.6]])
        labels = np.array([0])
        return features, labels


class TrainerWithoutDependencyInjection:
    def __init__(self) -> None:
        self.dataset_loader = ImageDatasetLoader()

    def train(self) -> None:
        features, labels = self.dataset_loader.load_feature_and_label()
        print(f"Training model on features: {features} and labels: {labels}")
```

This is a classic case of tight coupling, where the
`TrainerWithoutDependencyInjection` class is tightly coupled to the
`ImageDatasetLoader` class. This makes it difficult to test the
`TrainerWithoutDependencyInjection` class in isolation, as it's dependent on the
`ImageDatasetLoader` class and its behavior. If the `dataset_loader` is now an
instance of `TextDatasetLoader`, the `TrainerWithoutDependencyInjection` class
will fail to work as expected.

To fix this, we can use Dependency Injection to decouple the `Trainer` class
from the `DatasetLoader` class. This allows us to inject different
implementations of the `DatasetLoader` interface into the `Trainer` class,
making it easier to test and more flexible in terms of the data sources it can
work with.

```{code-cell} ipython3
class Trainer:
    def __init__(self, dataset_loader: DatasetLoader) -> None:
        self.dataset_loader = dataset_loader

    def train(self) -> None:
        features, labels = self.dataset_loader.load_feature_and_label()
        print(f"Training model on features: {features} and labels: {labels}")


def test_train_model_using_image_dataset_loader() -> None:
    dataset_loader = ImageDatasetLoader()
    trainer = Trainer(dataset_loader)
    trainer.train()


def test_train_model_using_text_dataset_loader() -> None:
    dataset_loader = TextDatasetLoader()
    trainer = Trainer(dataset_loader)
    trainer.train()
```

#### Stubs and Mocks

In unit testing, **mocks** and **stubs** are both types of test doubles used to
simulate the behavior of real objects in a controlled way. They are essential
tools for isolating the piece of code under test, ensuring that tests are fast,
reliable, and independent of external factors or system states. However, mocks
and stubs serve slightly different purposes and are used in different scenarios.

##### Stubs

**Stubs** provide predetermined responses to calls made during the test. They
are typically used to represent dependencies of the unit under test, allowing
you to bypass operations that are irrelevant to the test case, such as database
access, network calls, or complex logic. Stubs are simple objects that return
fixed data and are primarily used to:

-   Provide indirect input to the system under test.
-   Allow the test to control the test environment by simulating various
    conditions.
-   Avoid issues related to external dependencies, such as network latency or
    database access errors.

Stubs are passive and only return the specific responses they are programmed to
return, without any assertion on how they were used by the unit under test.

##### Mocks

**Mocks** are more sophisticated than stubs. They are used to verify the
interaction between the unit under test and its dependencies. Mocks can be
programmed with expectations, which means they can assert if they were called
correctly, how many times they were called, and with what arguments. They are
particularly useful for:

-   Verifying that the unit under test interacts correctly with its
    dependencies.
-   Ensuring that certain methods are called with the correct parameters.
-   Checking the number of times a dependency is interacted with, to validate
    the logic within the unit under test.

Mocks actively participate in the test, and failing to meet their expectations
will cause the test to fail. This makes them powerful for testing the behavior
of the unit under test.

#### Further Readings

-   [Unit Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/unit-testing/)
-   [Unit vs Integration vs System vs E2E Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/e2e-testing/testing-comparison/)

### Integration Testing

**Unlike unit testing**, which focuses on verifying the correctness of _isolated
pieces of code_, **integration testing** focuses on testing the **connections
and interactions between components** to identify any issues in the way they
integrate and operate together.

Verifying the **interactions between system components** is crucial, especially
since these components might be developed **independently or in isolation**. A
complex system typically encompasses **databases, APIs, interfaces**, and more,
all of which interact with each other and possibly with **external systems**.
**Integration testing** plays a key role in uncovering **system-wide issues**,
such as _inconsistencies in database schemas_ or _problems with third-party API
integrations_. It enhances overall **test coverage** and provides **vital
feedback** throughout the development process, ensuring that components work
together as intended[^unit-test-2].

#### Intuition

The analogy of a building can be extended to understand integration testing. If
unit tests verify the integrity of each brick (component), integration testing
checks the mortar between bricks (interactions). It ensures that not only are
the individual components reliable, but they also come together to form a
cohesive whole. Just as a wall relies on the strength of both the bricks and the
mortar, a software system relies on both its individual components and their
interactions.

Integration testing is like verifying that the electrical and plumbing systems
in a building work correctly once they are fitted together, despite each system
working perfectly in isolation.

#### Benefits of Integration Testing

##### Exposes Interface Issues

Integration testing is crucial for detecting problems that occur when different
parts of a system interact. It can uncover issues with the interfaces between
components, such as incorrect data being passed between modules, or problems
with the way components use each other's APIs.

##### Validates Functional Coherence

By testing a group of components together, integration testing ensures that the
software functions correctly as a whole. This is particularly important for
critical paths in an application where the interaction between components is
complex or involves external systems like databases or third-party services.

##### Highlights Dependency Problems

Complex systems often rely on external dependencies, and integration testing can
reveal issues with these dependencies that might not be apparent during unit
testing. This includes problems with network communications, database
integrations, and interactions with external APIs.

##### Improves Confidence in System Stability

Successful integration tests provide confidence that the system will perform as
expected under real-world conditions. This is especially important when changes
are made to one part of the system, as it helps ensure that such changes do not
adversely affect other parts.

Given the provided overview of integration testing, let's construct a clear and
practical guide to implementing integration testing, focusing on a hypothetical
banking application as mentioned. This guide will outline key steps,
considerations, and an example to illustrate how integration testing can be
effectively applied.

#### Understanding Integration Testing in Practice

Let's adopt the example given in
[Microsoft's Code with Engineering Playbook Integration Testing Design Blocks](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/#integration-testing-design-blocks)
to understand how integration testing can be applied in practice.

**Objective**: To ensure that independently developed modules of a banking
application—login, transfers, and current balance—work together as intended.

##### Step 1: Identify Integration Points

First, identify the key integration points within the application that require
testing. For the banking application, these points include:

-   **Login to Current Balance**: After a successful login, the application
    redirects the user to their current balance page with the correct balance
    displayed.
-   **Transfers to Current Balance**: After a transfer is initiated, ensure that
    the transfer completes successfully and the current balance is updated
    accurately.

##### Step 2: Design Integration Tests

For each identified integration point, design a test scenario that mimics
real-world usage:

-   **Login Integration Test**:

    -   **Objective**: Verify that upon login, the user is redirected to the
        current balance page with the correct balance.
    -   **Method**: Use a mock user with predefined credentials. After login,
        assert that the redirection is correct and the displayed balance matches
        the mock user's expected balance.

-   **Transfers Integration Test**:
    -   **Objective**: Confirm a transfer updates the sender's balance
        correctly.
    -   **Method**: Create a test scenario where a mock user transfers money to
        another account. Verify pre-transfer and post-transfer balances to
        ensure the transfer amount is correctly deducted.

Note that it is generally the consensus that integration tests should be using
real data and connections. So why did the example use the word "mock"? The
example is using the word "mock" to refer to the user, not the data or
connections. The user is a mock because it is not a real user, but a simulated
user for the purpose of testing.

#### Techniques for Integration Testing

##### Big Bang Testing

Big Bang Testing is a straightforward but high-risk approach to integration
testing where all the components or modules of a software application are
integrated simultaneously, and then tested as a whole. This method waits until
all parts of the system are developed and then combines them to perform the
integration test. The primary advantage of this approach is its simplicity, as
it does not require complex planning or integration stages. However, it has
significant drawbacks:

-   Identifying the root cause of a failure can be challenging because all
    components are integrated at once, making it difficult to isolate issues.
-   It can lead to delays in testing until all components are ready.
-   There's a higher risk of encountering multiple bugs or integration issues
    simultaneously, which can be overwhelming to debug and fix.

For example, if you want to test whether your `Trainer` class works correctly to
train a large language model, you might need to integrate the `Trainer` class
with the `LanguageModel` class, the `DatasetLoader` class, and the `Optimizer`
class. So here you would integrate all these classes at once and test the
`Trainer` class as a whole.

##### Incremental Testing

Incremental Testing is a more systematic and less risky approach compared to Big
Bang Testing. It involves integrating and testing components or modules one at a
time or in small groups. This method allows for early detection of defects
related to interfaces and interactions between integrated components.
Incremental Testing can be further divided into two main types: Top-Down Testing
and Bottom-Up Testing.

###### Top-Down Testing

Top-Down Testing involves integrating and testing from the top levels of the
software's control flow downwards. It starts with the highest-level modules and
progressively integrates and tests lower-level modules using stubs (simplified
implementations or placeholders) for modules that are not yet developed or
integrated. This approach allows for early validation of high-level
functionality and the overall system's architecture. However, it might delay
testing of lower-level components and their interactions.

Advantages include:

-   Early testing of major functionalities and user interfaces.
-   Facilitates early discovery of major defects.

Disadvantages include:

-   Lower-level modules are tested late in the cycle, which may delay the
    discovery of some bugs.
-   Requires the creation and maintenance of stubs, which can be
    resource-intensive.

###### Bottom-Up Testing

Bottom-Up Testing, in contrast, starts with the lowest level modules and
progressively moves up to higher-level modules, using drivers (temporary code
that calls a module and provides it with the necessary input for testing) until
the entire system is integrated and tested. This method is beneficial for
testing the fundamental components of a system early in the development cycle.

Advantages include:

-   Early testing of the fundamental operations provided by lower-level modules.
-   No need for stubs since testing begins with actual lower-level units.

Disadvantages include:

-   Higher-level functionalities and user interfaces are tested later in the
    development cycle.
-   Requires the development and maintenance of drivers, which can also be
    resource-intensive.

Both incremental approaches—Top-Down and Bottom-Up—offer more control and easier
isolation of defects compared to Big Bang Testing. They also allow for parallel
development and testing activities, potentially leading to more efficient use of
project time and resources.

#### Integration Test vs Acceptance Test

As we understand the importance of integration testing, which focuses on testing
the interactions between components, it is essential to distinguish integration
testing from acceptance testing. While both are critical for ensuring the
quality of a software system, they serve different purposes and operate at
different levels of the testing pyramid.

-   **Integration Testing**: Focuses on verifying the interactions between
    components to identify any issues in the way they integrate and operate
    together. It ensures that different parts of the system work together as
    intended from a technical perspective.
-   **Acceptance Testing**: Focuses on confirming a group of components work
    together as intended from a business scenario. It is performed by end-users
    or clients to validate the end-to-end business flow.

#### Further Readings

-   [Integration Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/)

### System Testing

#### Intuition

System testing can be likened to the inspection of a completed building before
it's opened for occupancy. After ensuring that all individual components
(bricks, electrical systems, plumbing) are working correctly and are properly
integrated, system testing examines the building as a whole to ensure it meets
all the specified requirements. This involves checking not only the internal
workings but also how the building interacts with external systems (such as
electrical grids, water supply systems) and complies with all applicable codes
and regulations. In software terms, system testing checks the complete and fully
integrated software product to ensure it aligns with the specified requirements.
It's about verifying that the entire system functions correctly in its intended
environment and meets all user expectations.

### End-to-End Testing

#### Intuition

End-to-end testing takes the building analogy a step further, comparing it to
not only inspecting the building as a whole but also observing how it serves its
occupants during actual use. Imagine a scenario where we follow residents as
they move in, live in, and use the building's various facilities. This would
include checking if the elevator efficiently transports people between floors,
if the heating system provides adequate warmth during winter, and if the
security systems ensure the residents' safety. In the context of software, E2E
testing involves testing the application's workflow from beginning to end. This
aims to replicate real user scenarios to ensure the system behaves as intended
in real-world use. It's the ultimate test to see if the software can handle what
users will throw at it, including interacting with other systems, databases, and
networks, to fulfill end-user requirements comprehensively.

Thus, while integration testing focuses on the connections and interactions
between components, system testing evaluates the complete, integrated system
against specified requirements, and end-to-end testing examines the system's
functionality in real-world scenarios, from the user's perspective.

### Unit vs Integration vs System vs E2E Testing

The table below illustrates the most critical characteristics and differences
among Unit, Integration, System, and End-to-End Testing, and when to apply each
methodology in a project.

|                         | Unit Test              | Integration Test                             | System Testing                                            | E2E Test                                                      |
| ----------------------- | ---------------------- | -------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| **Scope**               | Modules, APIs          | Modules, interfaces                          | Application, system                                       | All sub-systems, network dependencies, services and databases |
| **Size**                | Tiny                   | Small to medium                              | Large                                                     | X-Large                                                       |
| **Environment**         | Development            | Integration test                             | QA test                                                   | Production like                                               |
| **Data**                | Mock data              | Test data                                    | Test data                                                 | Copy of real production data                                  |
| **System Under Test**   | Isolated unit test     | Interfaces and flow data between the modules | Particular system as a whole                              | Application flow from start to end                            |
| **Scenarios**           | Developer perspectives | Developers and IT Pro tester perspectives    | Developer and QA tester perspectives                      | End-user perspectives                                         |
| **When**                | After each build       | After Unit testing                           | Before E2E testing and after Unit and Integration testing | After System testing                                          |
| **Automated or Manual** | Automated              | Manual or automated                          | Manual or automated                                       | Manual                                                        |

1. **Unit Testing**:

    - Tests individual units or components of the software in isolation (e.g.,
      functions, methods).
    - Ensures that each part works correctly on its own.

2. **Integration Testing**:

    - Tests the integration or interfaces between components or systems.
    - Ensures that different parts of the system work together as expected.

3. **System Testing**:

    - Tests the complete and integrated software system.
    - Verifies that the system meets its specified requirements.

4. **Acceptance Testing**:

    - Performed by end-users or clients to validate the end-to-end business
      flow.
    - Ensures that the software meets the business requirements and is ready for
      delivery.

5. **Regression Testing**:

    - Conducted after changes (like enhancements or bug fixes) to ensure
      existing functionalities work as before.
    - Helps catch bugs introduced by recent changes.

6. **Functional Testing**:

    - Tests the software against functional specifications/requirements.
    - Focuses on checking functionalities of the software.

7. **Non-Functional Testing**:

    - Includes testing of non-functional aspects like performance, usability,
      reliability, etc.
    - Examples include Performance Testing, Load Testing, Stress Testing,
      Usability Testing, Security Testing, etc.

8. **End-to-End Testing**:

    - Tests the complete flow of the application from start to end.
    - Ensures the system behaves as expected in real-world scenarios.

9. **Smoke Testing**:

    - Preliminary testing to check if the basic functions of the software work
      correctly.
    - Often done to ensure it's stable enough for further testing.

10. **Exploratory Testing**:

    - Unscripted testing to explore the application's capabilities.
    - Helps to find unexpected issues that may not be covered in other tests.

11. **Load Testing**:

    - Evaluates system performance under a specific expected load.
    - Identifies performance bottlenecks.

12. **Stress Testing**:

    - Tests the system under extreme conditions, often beyond its normal
      operational capacity.
    - Checks how the system handles overload.

13. **Usability Testing**:

    - Focuses on the user's ease of using the application, user interface, and
      user satisfaction.
    - Helps improve user experience and interface design.

14. **Security Testing**:

    - Identifies vulnerabilities in the software and ensures that the data and
      resources are protected.
    - Checks for potential exploits and security flaws.

15. **Compatibility Testing**:

    - Checks if the software is compatible with different environments like
      operating systems, browsers, devices, etc.

16. **Sanity Testing**:
    - A subset of regression testing, focused on testing specific
      functionalities after making changes.
    - Usually quick and verifies whether a particular function of the
      application is still working after a minor change.

### References and Further Readings

-   [Practical Test Pyramid - Martin Fowler](https://martinfowler.com/articles/practical-test-pyramid.html)
-   [Unit Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/unit-testing/)
-   [Integration Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/)
-   [Unit vs Integration vs System vs E2E Testing](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/e2e-testing/testing-comparison/)

## Phase 5. Continuous Deployment

Usually at this stage, we have a fully integrated and tested system that is
ready to be deployed from staging environment to production. The deployment
process is automated and follows a defined release process.

### Release

For example, we might have a `staging` environment that is a copy of the
production environment. We deploy to the `staging` environment first, and after
we have thoroughly tested the system, we deploy to the `production` environment.
This is an example of a release and can be easily done via workflow automation
like Github Actions, Gitlab CI, etc. If the deployed system is an application,
we can also deploy to container orchestration platforms like Kubernetes etc.

### Testing In Production

Machine learning systems are often non-deterministic, and it is not uncommon to
have good local evaluation results but poor production results. Consider a
machine learning system that predicts the price of a stock. You might get good
backtesting results, but when you deploy to production, you realize that the
model is not performing as expected due to various drifts.

We will follow Chip's book and mention a few techniques to test in production so
that we can still "rollback" to a previous version of the system if the
production results are not satisfactory.

#### Shadow Deployment/Mirrored Deployment

The mirrored deployment is a technique where we deploy both the existing model
$\mathcal{M}_1$ and the new model $\mathcal{M}_2$ to production. We do the
following:

1. Deploy both $\mathcal{M}_1$ and $\mathcal{M}_2$ to production.
2. For any incoming user requests, we will route to both $\mathcal{M}_1$ and
   $\mathcal{M}_2$ to make predictions, however, we will only use the
   predictions from the old model $\mathcal{M}_1$ for the final output to the
   users.
3. The new predictions are logged and persisted so the developers can analyze
   the results later.

By doing this, we can compare the results from the old model $\mathcal{M}_1$ and
the new model $\mathcal{M}_2$ to see if the new model is performing as expected.
If the new model is performing better, we can continue to use it. If the new
model is performing worse, we can rollback to the old model $\mathcal{M}_1$.

Of course one glaring issue is that this is expensive to run since we need to
serve twice, and this means inference costs are likely doubled.

#### A/B Testing

A/B Testing, also known as **Split Testing**, involves deploying two or more
versions of an application (Version A and Version B) simultaneously to different
segments of users. The primary objective is to compare the performance, user
engagement, and overall effectiveness of each version to inform data-driven
decisions about which version to fully roll out.

1. Deploy both the old model $\mathcal{M}_1$ and the new model $\mathcal{M}_2$
   to production.
2. Certain predictions are routed to $\mathcal{M}_1$ and certain predictions are
   routed to $\mathcal{M}_2$.
3. Based on predictions and user feedback, we can analyze the results and decide
   which model to keep.

##### 1. Randomized Traffic Allocation

Randomly assigning users to different versions (A and B) is essential to
eliminate bias and ensure that the test results are statistically valid.

-   **Equal Opportunity:** Randomization ensures that each user has an equal
    chance of being assigned to either version, which helps in controlling for
    variables that could otherwise skew results.
-   **Representative Sample:** This method creates a more representative sample
    of the overall user population, which helps in generalizing the results
    beyond the test group.
-   **Minimizes Confounding Variables:** Random assignment reduces the risk of
    confounding factors influencing the outcomes. For instance, if one version
    is shown to a certain demographic, the results might be biased due to
    inherent differences between that demographic and others.

##### 2. Sufficient Sample Size

Having a sufficiently large sample size is critical for obtaining statistically
significant results.

-   **Statistical Power:** A larger sample size increases the statistical power
    of the test, making it more likely to detect a true effect if one exists.
    This means you’re less likely to encounter Type II errors (failing to reject
    a false null hypothesis).
-   **Confidence Intervals:** Larger samples provide narrower confidence
    intervals, which gives a clearer picture of the effect size and its
    reliability. This helps in making informed decisions based on the results.
-   **Minimizes Variability:** With a larger sample, random variability is
    reduced, allowing the observed effects to be more accurately attributed to
    the changes being tested rather than random chance.

## Phase 6. Continuous Monitoring and Observability

### Motivation

Consider a financial institution that has a web-based banking application that
allows customers to transfer money, pay bills, and manage their accounts. The
motivation for implementing robust observing and monitoring practices is driven
by the critical need for **security**, **reliability**, **performance**, and
**regulatory compliance**.

-   Financial transactions are prime targets for **fraudulent activities** and
    **security breaches**. By **monitoring** system logs, network traffic, and
    user activities, the bank can identify and respond to potential security
    incidents in real time.
-   Customers expect banking services to be available 24/7, without
    interruptions. **Guaranteeing System Reliability and Availability** is
    therefore paramount to establish trust and confidence in your services.
    **Real-time health checks** and **performance metrics** can identify a
    failing server or an overloaded network segment, allowing IT teams to
    quickly reroute traffic or scale resources to prevent service disruption.

    In other words, Murphy's law is always at play, and things will fail at a
    certain point in time, and **you don't want to be oblivious to it**. If a
    system fails, we need to know it immediately and take action (logging and
    tracing are important to enable easy debugging).

### The What and The Why

We won't focus on the **how** to set up monitoring and observability, as there
are many ways to do it. For example,
[Grafana](https://grafana.com/blog/2023/11/20/ci-cd-observability-via-opentelemetry-at-grafana-labs/)
is one of the most popular open-source observability platforms, and it is
commonly used with
[Prometheus](https://grafana.com/docs/grafana/latest/getting-started/get-started-grafana-prometheus/),
a monitoring and alerting toolkit.

Instead, we need to give intuition on the **what** and **why** of monitoring and
observability.

```{list-table} Symptom and Cause
:header-rows: 1
:name: devops-ci-concept-monitoring-observability

*  - Symptom
   - Cause
*  - I’m serving HTTP 500s or 404s
   - Database servers are refusing connections
*  - My responses are slow
   - CPUs are overloaded by a bogosort, or an Ethernet cable is crimped under a rack, visible as partial packet loss
*  - Users in Antarctica aren’t receiving animated cat GIFs
   - Your Content Distribution Network hates scientists and felines, and thus blacklisted some client IPs
*  - Private content is world-readable
   - A new software push caused ACLs to be forgotten and allowed all requests
```

### The Four Golden Signals

```{admonition} Verbatim
:class: attention

The below section is verbatim from the
[Google SRE Book](https://sre.google/sre-book/monitoring-distributed-systems/).
```

The four golden signals of monitoring are latency, traffic, errors, and
saturation. If you can only measure four metrics of your user-facing system,
focus on these four.

#### Latency

The time it takes to service a request. It’s important to distinguish between
the latency of successful requests and the latency of failed requests. For
example, an HTTP 500 error triggered due to loss of connection to a database or
other critical backend might be served very quickly; however, as an HTTP 500
error indicates a failed request, factoring 500s into your overall latency might
result in misleading calculations. On the other hand, a slow error is even worse
than a fast error! Therefore, it’s important to track error latency, as opposed
to just filtering out errors.

#### Traffic

A measure of how much demand is being placed on your system, measured in a
high-level system-specific metric. For a web service, this measurement is
usually HTTP requests per second, perhaps broken out by the nature of the
requests (e.g., static versus dynamic content). For an audio streaming system,
this measurement might focus on network I/O rate or concurrent sessions. For a
key-value storage system, this measurement might be transactions and retrievals
per second.

#### Errors

The rate of requests that fail, either explicitly (e.g., HTTP 500s), implicitly
(for example, an HTTP 200 success response, but coupled with the wrong content),
or by policy (for example, "If you committed to one-second response times, any
request over one second is an error"). Where protocol response codes are
insufficient to express all failure conditions, secondary (internal) protocols
may be necessary to track partial failure modes. Monitoring these cases can be
drastically different: catching HTTP 500s at your load balancer can do a decent
job of catching all completely failed requests, while only end-to-end system
tests can detect that you’re serving the wrong content.

#### Saturation

How "full" your service is. A measure of your system fraction, emphasizing the
resources that are most constrained (e.g., in a memory-constrained system, show
memory; in an I/O-constrained system, show I/O). Note that many systems degrade
in performance before they achieve 100% utilization, so having a utilization
target is essential. In complex systems, saturation can be supplemented with
higher-level load measurement: can your service properly handle double the
traffic, handle only 10% more traffic, or handle even less traffic than it
currently receives? For very simple services that have no parameters that alter
the complexity of the request (e.g., "Give me a nonce" or "I need a globally
unique monotonic integer") that rarely change configuration, a static value from
a load test might be adequate. As discussed in the previous paragraph, however,
most services need to use indirect signals like CPU utilization or network
bandwidth that have a known upper bound. Latency increases are often a leading
indicator of saturation. Measuring your 99th percentile response time over some
small window (e.g., one minute) can give a very early signal of saturation.
Finally, saturation is also concerned with predictions of impending saturation,
such as "It looks like your database will fill its hard drive in 4 hours." If
you measure all four golden signals and page a human when one signal is
problematic (or, in the case of saturation, nearly problematic), your service
will be at least decently covered by monitoring.

### A Word on Monitoring in Machine Learning Systems

In the Machine Learning world, we may have to track things like model and data
shitfts. For example, model monitoring is about continuously tracking the
performance of models in production to ensure that they continue to provide
accurate and reliable predictions.

-   **Performance Monitoring**: Regularly evaluate the model's performance
    metrics in production. This includes tracking metrics like accuracy,
    precision, recall, F1 score for classification problems, or Mean Absolute
    Error (MAE), Root Mean Squared Error (RMSE) for regression problems, etc.

-   **Data Drift Monitoring**: Over time, the data that the model receives can
    change. These changes can lead to a decrease in the model's performance.
    Therefore, it's crucial to monitor the data the model is scoring on to
    detect any drift from the data the model was trained on.

-   **Model Retraining**: If the performance of the model drops or significant
    data drift is detected, it might be necessary to retrain the model with new
    data. The model monitoring should provide alerts or triggers for such
    situations.

-   **A/B Testing**: In case multiple models are in production, monitor their
    performances comparatively through techniques like A/B testing to determine
    which model performs better.

In each of these stages, it's essential to keep in mind principles like
reproducibility, automation, collaboration, and validation to ensure the
developed models are reliable, efficient, and providing value to the
organization.

## Phase 7. Continuous Learning and Training

As we consistently emphasized on monitoring and observing drifts in machine
learning systems, what would you do if you detect a drift? One common approach
is to re-train, or fine-tune the model on new data. So on top of _continuous
integration_, _continuous deployment_, and _continuous monitoring_, we also have
_continuous learning_ and _continuous training_.

Consider your object detection model that detects whether a person is wearing
safety helmets, masks and vests, but you notice that the model is performing
super poorly. It turns out the colors and the shapes of the objects are slightly
different now. So you likely need to collect more data, and retrain/fine-tune
the model on the new data.

```{admonition} See Also
:class: seealso

For more information on continuous learning and training, see Chapter 9.
Continual Learning and Test in Production of Chip Huyen's book, _Designing
Machine Learning Systems_.
```

## Appendix A. Styling, Formatting, and Linting

### Intuition

Guido Van Rossum, the author of Python, aptly stated, "Code is read more often
than it is written." This principle underscores the necessity of both clear
documentation and easy readability in coding. Adherence to style and formatting
conventions, particularly those based on
[PEP8](https://peps.python.org/pep-0008/), plays a vital role in achieving this
goal. Different teams may adopt various conventions, but the key lies in
consistent application and the use of automated pipelines to maintain this
consistency. For instance, standardizing line lengths simplifies code review
processes, making discussions about specific sections more straightforward. In
this context, **linting** and **formating** emerge as critical tools for
maintaining high code quality. Linting, the process of analyzing code for
potential errors, and formatting, which ensures a uniform appearance,
collectively boost **readability** and **maintainability**. A well-styled
codebase not only looks professional but also reduces bugs and eases
**integration** and **code reviews**. These practices, when ingrained as an
**intuition** among developers, lead to more robust and efficient software
development.

### Linting

#### Benefits of Linting

##### Code Quality Assurance

Linting tools, like [Pylint](https://github.com/pylint-dev/pylint) for Python,
automatically detect not just syntax errors but also a range of subtle issues
that could lead to bugs. This preemptive detection ensures higher **code
quality** and reliability.

In other words, linting tools help in catching potential issues early, reducing
the likelihood of bugs and errors in the codebase.

##### Reducing Technical Debt

In many big organizations, there is quality gate to pass if you were to deploy
your code to production. Henceforth, by catching potential issues early, linting
helps in reducing technical debt - the extra development work that arises from
choosing an easy solution now over a better approach that would take longer.

##### Maintainability and Scalability

Linting enforces readability and uniformity, making the codebase easier to
understand and modify. This is crucial for long-term **maintenance** and
**scaling** of the project.

#### The PEP8 Standard

The [PEP8 guide](https://peps.python.org/pep-0008/) offers essential coding
conventions.

##### Simple Styling Practices

1. **Indentation**: PEP8 recommends 4 spaces per indentation level. While Python
   is flexible with indentation size (any consistent `k` spaces), adhering to
   the 4-space convention promotes uniformity across the Python community.

2. **Line Length**: A suggested maximum of 79 characters per line enhances
   readability, especially in environments without dynamic wrapping. Different
   organizations may vary, but consistency is key. This limit roots in
   historical constraints and current practicality.

3. **Variable Naming**: Readability is crucial. Variables should be descriptive,
   making code understandable at a glance. For example, `name = "John"` is more
   descriptive than `a = "John"`, as it clearly indicates the variable's
   purpose.

4. **Import Statements**: Avoid wildcard imports like `from .src.main import *`.
   They obscure the origin of functions, complicating maintenance and
   readability. A more complex issue arises with relative imports in a deeply
   nested package structure, which can lead to confusion about the source file's
   location and dependencies.

##### A More Nuanced Example: Mutable Default Arguments

The issue of mutable default arguments in Python demonstrates a subtle yet
significant trap that we encounter. Consider a function `add_to_list` designed
to append an item to a list. When using a default mutable argument like an empty
list (`[]`), the list isn't reinitialized on each function call. This results in
unexpected behavior, where subsequent calls to the function without specifying a
list continue to add items to the same list.

To address this, a better practice is to use `None` as the default argument.
Inside the function, check if the argument is `None` and, if so, initialize a
new list. This ensures that each function call operates on a fresh list unless
otherwise specified.

Let's see this in action.

Consider the following code snippet:

```{code-cell} ipython3
from __future__ import annotations

from typing import List, TypeVar

T = TypeVar('T')

def add_to_list(item: T, some_list: List[T] = []) -> List[T]:
    some_list.append(item)
    return some_list
```

This looks harmless, but if you run the below code, you will see that the
function does not behave as expected:

```{code-cell} ipython3
list_1 = add_to_list(0)  # [0]
print(f"list_1: {list_1}")

list_2 = add_to_list(1)  # [0, 1]
print(f"list_2: {list_2}")

print(f"list_1: {list_1}") # [0, 1]
```

Why did `list_2` not return `[1]`? The issue lies in the default argument
`some_list: List[T] = []`. This default argument is evaluated only once, when
the function is defined, and not every time the function is called. This means
that the same list is used every time the function is called without the
`some_list` argument. This means that if you use a mutable default argument and
mutate it, you will and have mutated that object for all future calls to the
function as well. And if you print `list_1` again after `list_2`, you will see
that `list_1` has also been mutated!

To fix this, you can use `None` as the default argument and then initialize the
list inside the function:

```{code-cell} ipython3
def add_to_list(item: T, some_list: List[T] | None = None) -> List[T]:
    if some_list is None:
        some_list = []
    some_list.append(item)
    return some_list
```

Then the function will behave as expected:

```{code-cell} ipython3
list_1 = add_to_list(0)  # [0]
list_2 = add_to_list(1)  # [1]
print(f"list_1: {list_1}")
print(f"list_2: {list_2}")
```

If this goes into **production**, it could lead to a **bug** that is _hard_ to
catch.

#### Tools

In the industry, there are a few leading tools:

-   [Pylint](https://pylint.pycqa.org/en/latest/index.html)
-   [Flake8](https://flake8.pycqa.org/en/latest/)
-   [Ruff](https://docs.astral.sh/ruff/)

Ruff is a new entrant in the market and is gaining popularity due to its speed
as it is written in Rust. Whichever the tool the team choose, the key is to
ensure that it is integrated into the development workflow consistently.

#### Best Practices for Linting and CI/CD Pipeline Integration

##### Automate Linting and Integration with CI/CD Pipelines

Integrating linting into a CI/CD pipeline typically involves the following
steps:

1. **Configuration**: Define linting rules in a configuration file. This file is
   then placed in the project repository.

2. **Pipeline Setup**: In the CI/CD system, create a job or stage specifically
   for linting. This job will execute whenever a new commit is pushed to the
   repository.

3. **Running Linter**: During the linting stage, the CI/CD system runs the
   linter against the codebase using the defined rules.

4. **Handling Linting Results**: If the linter finds issues, it can fail the
   build, preventing further stages (like testing or deployment) until the
   issues are resolved.

5. **Feedback to Developers**: The results of the linting process are reported
   back to the developers, usually through the CI/CD system's interface or via
   notifications.

This integration ensures that code quality checks are an automated and
consistent part of the development cycle.

##### Local and CI Environment Consistency

The remote Continuous Integration (CI) environment is a safety net and quality
gate for the codebase. This does not mean you should wait for feedback from the
CI environment to fix issues.

Why? Imagine you committed a large piece of code without any regards to the
linting rules. The CI environment will fail, and you will have to fix the issues
and push the code again. Then again, there is no guarantee that the CI
environment will pass. This is a waste of time and resources.

What should you do? You should lint your code locally before pushing it to the
remote repository. This will ensure that the CI environment will pass, and you
will not have to wait for feedback from the CI environment.

Consequently, it is essential to maintain consistency between the local
development environment and the CI environment. This consistency ensures that
the code behaves consistently across different setups. In other words, the lint
rules defined locally should be the same as those defined in the CI environment.
One source of truth is the mantra that should be followed, if not, a rule
defined in CI which is not defined locally may fail the build. Conversely, a
locally defined rule A might not be defined in the CI environment, leading to a
false sense of security.

##### Pre-Commit Hooks

Pre-commit hooks are scripts that run before a commit is made. They are a
powerful tool for ensuring that code quality checks are performed before
commits. This can include linting, formatting, and other checks such as testing.
This is a good to have as it injects some sort of discipline and automation into
the local development environment.

##### Order in Pipeline

In a CI/CD pipeline, the typical sequence is to lint first, then format, and
finally run unit tests. Linting first helps catch syntax errors and code smells
early, reducing the likelihood of these issues causing test failures. Formatting
next ensures code consistency, and finally, unit tests validate the
functionality. This order optimizes the build process, catching errors
efficiently and maintaining code quality.

### Formatting

What is formatting? Formatting is the process of ensuring that the codebase
adheres to a consistent style/format. This includes indentation, line length and
spacing, among other things. The goal is to make the codebase more readable and
maintainable. This will reduce friction in code reviews. Imagine the frustration
if developer A uses a 120 character line length and developer B uses 80
characters. They will not be in sync with each other.

#### What is the Difference between Linting and Formatting?

The difference might be nuanced and isn't clear. The tagline, **linters for
catching errors and quality, formatters to fix code formatting style** can be
demonstrated with an example:

```python
from typing import List, TypeVar

T = TypeVar('T')

def add_to_list_and_purposely_make_the_list_very_very_very_long(item: T, some_list: List[T] = []) -> List[T]:
    some_list.append(item)
    return some_list
```

-   Our linter will complain something like "Mutable default argument" as this
    is a potential bug. This is where our linter such as `ruff` or `pylint` will
    come into play. The linter will suggest to you to take action but won't take
    action for you. Furthermore, a formatter such as `black` won't catch this
    issue because they are not designed to catch such issues.

-   Our linter and formatter will also see another glaring issue, that is the
    `if` line is too long, exceeding the `PEP8` standard of $79$ length. Both
    `black` and `ruff` will tell us this, but `black` will perform an
    **in-place** treatment, formatting the code on the go for you, whereas
    `ruff` will just tell you.

Therefore, the coding world generally uses a formatter (`black`) and a linter
(`ruff`) in tandem.

#### Tools

In the industry, there are a few leading tools:

-   [Black](https://black.readthedocs.io/en/stable/)
-   [Ruff](https://docs.astral.sh/ruff/)

Interestingly, `ruff` serves as both a linter and a formatter, so we can have an
all in one package. However, `black` seems to be the most popular formatter in
the Python and generally, more matured.

Many teams also add in `isort` to sort the imports. This is a good practice as
it makes the imports more readable.

#### Best Practices for Formatting and CI/CD Pipeline Integration

In general, the best practices for formatting are similar to those for linting.
The key is to ensure that the formatting tool is integrated into the development
workflow consistently.

### Where to Start?

-   [PyTorch](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
-   [OpenAI](https://github.com/openai/openai-python/blob/main/pyproject.toml)
-   [FastAPI](https://github.com/tiangolo/fastapi/blob/master/pyproject.toml)

### References and Further Readings

-   [Code Style Checks - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/continuous-integration/#code-style-checks)
-   [Code Analysis Linting - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/code-reviews/recipes/python/#code-analysis-linting)
-   [Differences between code linters and formatters](https://taiyr.me/what-is-the-difference-between-code-linters-and-formatters)
-   [Format Code vs Lint Code](https://medium.com/@awesomecode/format-code-vs-and-lint-code-95613798dcb3)
-   [PEP8 guide](https://peps.python.org/pep-0008/)
-   [Pre-commits Styling](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)

## References and Further Readings

-   [Welcome to pre-commit heaven - Marvelous MLOps Substack](https://marvelousmlops.substack.com/i/130911126)
-   [MLOps Basics [Week 6]: CI/CD - GitHub Actions](https://www.ravirajag.dev/blog/mlops-github-actions)
-   [CI/CD for Machine Learning](https://madewithml.com/courses/mlops/cicd/)
-   [Stop saying "technical debt" - Stack Overflow](https://stackoverflow.blog/2023/12/27/stop-saying-technical-debt/)
-   [Is Python strongly typed? - Stack Overflow](https://stackoverflow.com/questions/11328920/is-python-strongly-typed)
-   [Chapter 6. Monitoring Distributed Systems - Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/)
-   "Chapter 9. Continual Learning and Test in Production." In Designing Machine
    Learning Systems: An Iterative Process for Production-Ready Applications,
    O'Reilly Media, Inc., 2022.

[^stop_saying_technical_debt]:
    [Stop saying "technical debt" - Stack Overflow](https://stackoverflow.blog/2023/12/27/stop-saying-technical-debt/)

[^python_strongly_and_dynamic_typing]:
    [Is Python strongly typed? - Stack Overflow](https://stackoverflow.com/questions/11328920/is-python-strongly-typed)

[^google-sre-monitoring]:
    [Chapter 6. Monitoring Distributed Systems - Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/)

[^1]:
    [Deliver Quickly and Daily - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/continuous-integration/#deliver-quickly-and-daily)

[^2]: [Common Gotchas](https://docs.python-guide.org/writing/gotchas/)
[^unit-test-1]:
    [Why Unit Tests - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/unit-testing/why-unit-tests/)

[^unit-test-2]:
    [Why Integration Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/#why-integration-testing)

[^unit-test-3]:
    [Unit vs Integration vs System vs E2E Testing](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/e2e-testing/testing-comparison/)
