# DevOps Best Practices

## GitHub Actions

Things discussed here are concepts that are not specific to GitHub Actions. It
can be replicated in other CI tools such as Travis CI, CircleCI, Jenkins, etc
with some modifications of syntax.

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
