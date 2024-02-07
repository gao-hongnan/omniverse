# DevOps Best Practices

## Pre-commit Setup Guide

Pre-commit is a framework that manages and maintains multi-language pre-commit
hooks. It simplifies code quality checks and ensures standards are met before
commits are made.

### Step 1: Install Pre-commit

Install pre-commit on your system. For most users, using pip is sufficient:

```bash
pip install pre-commit
```

### Step 2: Create a Configuration File

Create a `.pre-commit-config.yaml` file at the root of your repository. This
file will contain the configuration for your pre-commit hooks.

Example:

```yaml
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v3.4.0
      hooks:
          - id: trailing-whitespace
          - id: end-of-file-fixer
```

### Step 3: Install Hooks

Run the following command to install the hooks defined in your
`.pre-commit-config.yaml`:

```bash
pre-commit install
```

This command sets up the pre-commit hooks in your local repository.

### Step 4: Run Against All Files

Optionally, you can run the hooks against all the files in your repository to
see if they pass:

```bash
pre-commit run --all-files
```

### Step 5: Make a Commit

Once the hooks are installed, they will automatically run on every `git commit`.
If a hook fails, it will block the commit. You can fix the issues and try
committing again.

## GitHub Actions

Things discussed here are concepts that are not specific to GitHub Actions. It
can be replicated in other CI tools such as Travis CI, CircleCI, Jenkins, etc
with some modifications of syntax.

The
[PyTorch GitHub Actions](https://github.com/pytorch/pytorch/blob/main/.github)
has a comprehensive set of workflows that can be used as a reference.

### Environment Variables

#### `WORKDIR`

```yaml
WORKDIR:
    ${{ inputs.working-directory == '' && '.' || inputs.working-directory }}
```

-   **`inputs.working-directory`**: This refers to an input parameter of the
    workflow. When the workflow is called, a value can be passed to this input,
    specifying the directory from which certain actions within the workflow
    should be executed.

-   **Conditional Assignment**:

    -   The expression
        `${{ inputs.working-directory == '' && '.' || inputs.working-directory }}`
        is a conditional assignment. It checks if `inputs.working-directory` is
        an empty string (`''`).
    -   If it is empty (`inputs.working-directory == ''`), then `WORKDIR` is set
        to `'.'`, which represents the current directory (typically the root of
        the GitHub repository).
    -   If `inputs.working-directory` is not empty, `WORKDIR` is set to whatever
        value `inputs.working-directory` holds.

-   **Purpose of `WORKDIR`**:

    -   `WORKDIR` is used as an environment variable within the GitHub Actions
        runner environment. It can be referenced by subsequent steps in the
        workflow to determine the working directory for various operations.
    -   By defining `WORKDIR` this way, the workflow allows for dynamic setting
        of the working directory based on the input parameter, providing
        flexibility in how the workflow is used.

-   **Usage in Workflow Steps**:
    -   In the steps of the workflow, you might see commands that use `WORKDIR`,
        like `cd $WORKDIR`. This would change the current directory to the one
        specified by `inputs.working-directory`, or to the repository root if no
        directory was specified.

In summary, `WORKDIR` in this context is a custom environment variable used to
dynamically set the working directory for the GitHub Actions workflow based on
an input parameter. This approach allows the workflow to adapt to different
directory structures or requirements based on how it's triggered.

## Continuous Integration

### Pinning DevOps Tool Versions

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

#### Example: Pinning `pylint` Version

Consider the Python linting tool `pylint`. It's known for its thoroughness in
checking code quality and conformity to coding standards. However, `pylint` is
also frequently updated, with new releases potentially introducing new checks or
modifying existing ones.

Suppose your project is using `pylint` version 2.6.0. In this version, your
codebase passes all linting checks, ensuring a certain level of code quality and
consistency. Now, imagine `pylint` releases a new version, 2.7.0, which includes
a new check for a particular coding pattern (e.g., enforcing more stringent
rules on string formatting or variable naming).

##### Scenario Without Pinning

-   Consider the case where you have unpinned version `pylint` in your build.
    There's a good chance that your CI environment (which uses your build)
    automatically updates to `pylint` 2.7.0, this new check might suddenly cause
    your build to fail, even though there have been no changes in your codebase.
    This can disrupt development workflows and require immediate attention to
    fix linting errors, which might not align with your current development
    priorities.

##### Scenario With Pinning

-   By pinning `pylint` to version 2.6.0 in your CI pipeline, you ensure that
    your code is always tested against this specific version. This provides
    stability and predictability, as you won't encounter sudden build failures
    due to new linting rules introduced in newer versions of `pylint`.

-   When you're ready to upgrade to `pylint` 2.7.0, you can do so intentionally.
    This allows you to allocate time to address any new linting errors that
    arise from the update, ensuring a controlled and manageable update process.

##### Implementing Pinning

In your `requirements.txt` or equivalent dependency file, instead of having:

```plaintext
pylint
```

You would specify:

```plaintext
pylint==2.6.0
```

This change ensures that `pylint` 2.6.0 is used in your CI environment,
regardless of newer versions being available.

#### Conclusion

In summary, pinning versions in CI is important for ensuring consistency,
stability, and security, but it requires a balance with maintenance effort to
ensure tools don’t become outdated. Regularly scheduled reviews and updates of
these pinned versions can help maintain this balance.

### Linter

One source of truth between the CI environment and the development environment.

-   [PyTorch](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
-   [OpenAI](https://github.com/openai/openai-python/blob/main/pyproject.toml)
-   [FastAPI](https://github.com/tiangolo/fastapi/blob/master/pyproject.toml)

Check whether we need black formatting options in ruff if we already have one to
use for black?

## Building the book

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

---

# CI

-   [DevOps Best Practices](#devops-best-practices)
    -   [Pre-commit Setup Guide](#pre-commit-setup-guide)
        -   [Step 1: Install Pre-commit](#step-1-install-pre-commit)
        -   [Step 2: Create a Configuration File](#step-2-create-a-configuration-file)
        -   [Step 3: Install Hooks](#step-3-install-hooks)
        -   [Step 4: Run Against All Files](#step-4-run-against-all-files)
        -   [Step 5: Make a Commit](#step-5-make-a-commit)
    -   [GitHub Actions](#github-actions)
        -   [Environment Variables](#environment-variables)
            -   [`WORKDIR`](#workdir)
    -   [Continuous Integration](#continuous-integration)
        -   [Pinning DevOps Tool Versions](#pinning-devops-tool-versions)
            -   [Example: Pinning `pylint` Version](#example-pinning-pylint-version)
                -   [Scenario Without Pinning](#scenario-without-pinning)
                -   [Scenario With Pinning](#scenario-with-pinning)
                -   [Implementing Pinning](#implementing-pinning)
            -   [Conclusion](#conclusion)
        -   [Linter](#linter)
    -   [Building the book](#building-the-book)
    -   [How to Index Jupyter Book?](#how-to-index-jupyter-book)
-   [CI](#ci)
    -   [Motivation](#motivation)
        -   [Introduction to the Problem](#introduction-to-the-problem)
        -   [Broader Implications](#broader-implications)
        -   [Impact on Development Workflow](#impact-on-development-workflow)
        -   [Long-Term Consequences](#long-term-consequences)
        -   [Advocating for a Solution](#advocating-for-a-solution)
        -   [Enhancing Resilience and Reducing Bugs with Linting, Testing, and Static Type Checking](#enhancing-resilience-and-reducing-bugs-with-linting-testing-and-static-type-checking)
            -   [Introduction: The Role of Quality Assurance Tools](#introduction-the-role-of-quality-assurance-tools)
            -   [Advantages of Automated Testing](#advantages-of-automated-testing)
            -   [Introduction: Python's Typing Landscape](#introduction-pythons-typing-landscape)
            -   [Role of Static Type Checking](#role-of-static-type-checking)
            -   [Reducing Runtime Errors](#reducing-runtime-errors)
            -   [Conclusion: Building a Resilient Codebase](#conclusion-building-a-resilient-codebase)
    -   [Version Control](#version-control)
    -   [Code Quality and Standards](#code-quality-and-standards)
    -   [Dependency Management](#dependency-management)
    -   [Testing](#testing)
    -   [Continuous Integration (CI)](#continuous-integration-ci)
    -   [Continuous Deployment/Delivery (CD)](#continuous-deploymentdelivery-cd)
    -   [Monitoring and Logging](#monitoring-and-logging)
    -   [Security](#security)
    -   [Infrastructure](#infrastructure)
    -   [Documentation](#documentation)
    -   [Others](#others)
    -   [Best Practices](#best-practices)
    -   [Continuous Improvement](#continuous-improvement)

## Motivation

### Introduction to the Problem

-   **Immediate Challenge**:When cloning repositories from previous team
    members, we often find that the code fails to pass basic checks. This not
    only hinders immediate local development but also fails to integrate with
    our Jenkins automated process.

    Furthermore, we will not have time and resource to correct tens of thousands
    of lines of code. This is **technical debt**. What are the broader
    implications?s

### Broader Implications

-   **Quality and Reliability Concerns**: The code failing such checks is often
    symptomatic of deeper quality issues, potentially leading to unreliable and
    unstable applications. Unchecked code is more likely to contain bugs,
    security vulnerabilities, or performance issues.

    This is **technical debt** and often results in code breaking during
    **production**. One may ask, even if we do the checking of quality, it does
    not 100% reduce the occurence of bugs. That is 100% correct, but the
    argument here is not to **eradicate**, but to **reduce** - a numbers game.

### Impact on Development Workflow

**Technical debt** again. Why? Because if the new developer comes, he faces:

-   **Increased Onboarding Time**: Highlight that new team members or those
    inheriting the code spend unnecessary time fixing these basic issues,
    delaying their ability to contribute effectively.
-   **Inefficient Use of Resources**: Point out that developers often spend a
    significant amount of time troubleshooting and resolving issues that could
    have been easily prevented with proper checks in place.

He will also need to decide with the team if it is worth refactoring potentially
tens of thousands of code!!!

### Long-Term Consequences

-   **Technical Debt Accumulation**: Emphasize that allowing such code practices
    creates and accumulates technical debt. Over time, this debt becomes harder
    to address, leading to a slower development cycle and increased costs.
-   **Scalability and Maintenance Issues**: Stress that as the codebase grows,
    maintaining code that hasn't been properly vetted for quality becomes
    increasingly challenging, leading to scalability issues.
-   **Erosion of Best Practices**: Mention that neglecting these checks sets a
    precedent that might lead to a gradual erosion of coding standards and best
    practices within the team.

### Advocating for a Solution

**A good example
[here](https://ci4.corp.dbs.com:8443/job/CBBA_REGIONAL/job/synthetic-data-generation/job/feature%252Fmultitable/).
But it is missing static type check I think - everybody is scared to fix typing
issues in python but look at how big companies do it. It enforced good code
hygience. **

![](cicd-sample.PNG)

-   **Need for Stringent Checks**: Advocate for stringent linting, testing, and
    static type checking as part of your CI/CD pipeline. This ensures that all
    code, regardless of its origin, meets a certain quality standard before it's
    integrated.
-   **Preventing Future Issues**: Argue that by enforcing these checks, future
    teams will inherit healthier, more robust codebases, reducing the time and
    effort needed to get up to speed and start contributing effectively.

Certainly! The section you're looking to write focuses on the benefits of
linting, testing, and static type checking in enhancing resilience and reducing
bugs in production. These tools are crucial in a robust development process
because they help identify potential issues early in the development cycle, thus
preventing them from becoming more significant problems in production. Here's a
structured way to articulate this:

### Enhancing Resilience and Reducing Bugs with Linting, Testing, and Static Type Checking

#### Introduction: The Role of Quality Assurance Tools

-   **Overview**: Linting, testing, and static type checking are foundational
    tools in a developer's arsenal to ensure code quality, resilience, and
    reliability.

#### Advantages of Automated Testing

-   **Catching Bugs Early**: Automated tests, including unit and integration
    tests, are designed to catch bugs at the earliest possible stage. This early
    detection prevents bugs from propagating to later stages of development or
    into production.
-   **Ensuring Code Functionality**: Tests validate that the code performs as
    expected, reducing the likelihood of functional errors in production.
-   **Facilitating Refactoring and Updates**: With a robust test suite,
    developers can confidently refactor and update code, knowing that tests will
    catch any inadvertent breakages or regressions.

#### Introduction: Python's Typing Landscape

-   **Dynamic Typing in Python**: While Python's dynamic typing allows for rapid
    development and iteration, it can also introduce subtle bugs, especially in
    larger or more complex codebases.

#### Role of Static Type Checking

-   **Type Safety**: Static type checkers ensure that variables and functions
    are used correctly according to their defined types, preventing type-related
    errors that can be hard to trace in dynamically typed languages like Python.
-   **Catching Type Errors Early**: Static type checking tools like `mypy`
    analyze code for type consistency before it is run. This early detection of
    type mismatches prevents runtime type errors that could otherwise occur in
    production.
-   **Improved Code Readability and Maintainability**: Type annotations make the
    code more readable and self-documenting. They clarify the intended use of
    variables, functions, and classes, making the codebase easier to understand
    and maintain, especially for teams.
-   **Facilitating Better Code Design**: The practice of adding type hints
    encourages developers to think more carefully about their data structures
    and function interfaces, often leading to better designed, more robust code.

#### Reducing Runtime Errors

-   **Preventing Common Python Bugs**: Type checking helps to prevent common
    bugs in Python related to incorrect type usage, such as passing an incorrect
    data type to a function or incorrectly handling return types.
-   **Enhancing Safety in Refactoring**: When modifying existing code, type
    annotations provide an extra layer of safety, ensuring that changes do not
    inadvertently alter the expected type behavior of the code.

#### Conclusion: Building a Resilient Codebase

-   **Proactive Approach to Quality**: Summarize by emphasizing that integrating
    linting, testing, and static type checking into the development process is a
    proactive approach to building a resilient and robust codebase.
-   **Reducing Production Bugs and Downtime**: Highlight that these tools
    significantly reduce the likelihood of bugs reaching production, thereby
    minimizing downtime and the associated costs.
-   **Long-Term Benefits**: Conclude by mentioning the long-term benefits: a
    healthier codebase, smoother development process, and increased confidence
    in the code's reliability and performance.

---

## Version Control

-   **Git** with a platform like GitHub, GitLab, or Bitbucket.
-   **Branching Strategy**: Implement a strategy like Git Flow or Trunk Based
    Development.

## Code Quality and Standards

-   **Linting**: Use tools like `flake8` or `pylint`.
-   **Code Formatting**: Tools like `black` or `autopep8`.
-   **Type Checking**: Optional but recommended, using `mypy` or `pytype`.
-   **Code Reviews**: Enforce via merge/pull request policies.

## Dependency Management

-   **Dependency Specification**: Use `pyproject.toml` (PEP 518) to define
    dependencies.
-   **Virtual Environments**: Use `venv`, `pipenv`, or `poetry` for isolated
    environments.
-   **Dependency Scanning**: Tools like `snyk` or `dependabot` for vulnerability
    scanning.

## Testing

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

## Continuous Integration (CI)

-   **Automated Builds**: On every commit/merge to main branches.
-   **Automated Tests**: Run all tests on build.
-   **Quality Gates**: No merge if tests, linting, or type checks fail.
-   **Docker Integration**: For packaging and consistent test environments.

## Continuous Deployment/Delivery (CD)

-   **Automated Deployment**: Use tools like Jenkins, GitLab CI/CD, GitHub
    Actions, or CircleCI.
-   **Environment Management**: Dev, Staging, and Production environments with
    promotion strategy.
-   **Infrastructure as Code (IaC)**: Tools like Terraform or AWS
    CloudFormation.
-   **Configuration Management**: Tools like Ansible, Chef, or Puppet.

## Monitoring and Logging

-   **Application Performance Monitoring (APM)**: Tools like New Relic, Datadog,
    or Prometheus.
-   **Logging**: Centralized logging with ELK stack (Elasticsearch, Logstash,
    Kibana) or similar.
-   **Error Tracking**: Tools like Sentry.

## Security

-   **Static Application Security Testing (SAST)**: Tools like Bandit for Python
    code.
-   **Dynamic Application Security Testing (DAST)**: Automated scanning of web
    applications.
-   **Secrets Management**: Avoid hardcoding secrets, use tools like HashiCorp
    Vault or AWS Secrets Manager.

## Infrastructure

-   **Containerization**: Docker for packaging and Kubernetes for orchestration.
-   **Cloud Providers**: AWS, GCP, Azure, or a combination for cloud
    infrastructure.

## Documentation

-   **Code Documentation**: Inline comments and external documentation (e.g.,
    Sphinx for Python).
-   **Pipeline Documentation**: Document the CI/CD process, setup, and any
    manual steps required.

## Others

-   **Feature Flags**: For controlled rollouts of new features.
-   **Rollback Strategies**: Automated rollback in case of failed deployments.
-   **Load Testing and Stress Testing**: Regularly test the system's
    performance.

## Best Practices

-   **Immutable Artifacts**: Build once and promote the same artifact through
    environments.
-   **Blue/Green or Canary Deployments**: For zero-downtime deployments and A/B
    testing.
-   **Microservices Architecture**: If applicable, for better scalability and
    maintenance.

## Continuous Improvement

-   **Regular Review of Tools and Processes**: Stay updated with evolving best
    practices and tools.
-   **Feedback Loops**: Implement mechanisms for continuous feedback from all
    stakeholders.

Remember, the exact tools and practices may vary depending on the specific
requirements, team size, and existing infrastructure of your organization. The
key is to maintain a balance between rigorous quality controls and efficient
development workflows.
