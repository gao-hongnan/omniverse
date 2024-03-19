# Continuous Integration (CI) Workflow

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Maybe_Chaotic-orange)

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

## Phase 4. Scan and Test

...

## Phase 5. Continuous Deployment

### Release

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

### The What and Why.

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

## References and Further Readings

-   [Welcome to pre-commit heaven - Marvelous MLOps Substack](https://marvelousmlops.substack.com/i/130911126)
-   [MLOps Basics [Week 6]: CI/CD - GitHub Actions](https://www.ravirajag.dev/blog/mlops-github-actions)
-   [CI/CD for Machine Learning](https://madewithml.com/courses/mlops/cicd/)
-   [Stop saying "technical debt" - Stack Overflow](https://stackoverflow.blog/2023/12/27/stop-saying-technical-debt/)
-   [Is Python strongly typed? - Stack Overflow](https://stackoverflow.com/questions/11328920/is-python-strongly-typed)
-   [Chapter 6. Monitoring Distributed Systems - Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/)

[^stop_saying_technical_debt]:
    [Stop saying "technical debt" - Stack Overflow](https://stackoverflow.blog/2023/12/27/stop-saying-technical-debt/)

[^python_strongly_and_dynamic_typing]:
    [Is Python strongly typed? - Stack Overflow](https://stackoverflow.com/questions/11328920/is-python-strongly-typed)

[^google-sre-monitoring]:
    [Chapter 6. Monitoring Distributed Systems - Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/)
