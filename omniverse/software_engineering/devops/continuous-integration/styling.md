# Styling, Formatting, and Linting

```{contents}
:local:
```

## Intuition

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

## Linting

### Benefits of Linting

#### Code Quality Assurance

Linting tools, like [Pylint](https://github.com/pylint-dev/pylint) for Python,
automatically detect not just syntax errors but also a range of subtle issues
that could lead to bugs. This preemptive detection ensures higher **code
quality** and reliability.

In other words, linting tools help in catching potential issues early, reducing
the likelihood of bugs and errors in the codebase.

#### Reducing Technical Debt

In many big organizations, there is quality gate to pass if you were to deploy
your code to production. Henceforth, by catching potential issues early, linting
helps in reducing technical debt - the extra development work that arises from
choosing an easy solution now over a better approach that would take longer.

#### Maintainability and Scalability

Linting enforces readability and uniformity, making the codebase easier to
understand and modify. This is crucial for long-term **maintenance** and
**scaling** of the project.

### The PEP8 Standard

The [PEP8 guide](https://peps.python.org/pep-0008/) offers essential coding
conventions.

#### Simple Styling Practices

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

#### A More Nuanced Example: Mutable Default Arguments

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

```python
from typing import List, TypeVar

T = TypeVar('T')

def add_to_list(item: T, some_list: List[T] = []) -> List[T]:
    some_list.append(item)
    return some_list
```

This looks harmless, but if you run the below code, you will see that the
function does not behave as expected:

```python
list_1 = add_to_list(0)  # [0]
list_2 = add_to_list(1)  # [0, 1]
```

Why did `list_2` not return `[1]`? The issue lies in the default argument
`some_list: List[T] = []`. This default argument is evaluated only once, when
the function is defined, and not every time the function is called. This means
that the same list is used every time the function is called without the
`some_list` argument. To fix this, you can use `None` as the default argument
and then initialize the list inside the function:

```python
def add_to_list(item: T, some_list: List[T] | None = None) -> List[T]:
    if some_list is None:
        some_list = []
    some_list.append(item)
    return some_list
```

Then the function will behave as expected:

```python
list_1 = add_to_list(0)  # [0]
list_2 = add_to_list(1)  # [1]
```

If this goes into **production**, it could lead to a **bug** that is _hard_ to
catch.

### Tools

In the industry, there are a few leading tools:

-   [Pylint](https://pylint.pycqa.org/en/latest/index.html)
-   [Flake8](https://flake8.pycqa.org/en/latest/)
-   [Ruff](https://docs.astral.sh/ruff/)

Ruff is a new entrant in the market and is gaining popularity due to its speed
as it is written in Rust. Whichever the tool the team choose, the key is to
ensure that it is integrated into the development workflow consistently.

### Best Practices for Linting and CI/CD Pipeline Integration

#### Automate Linting and Integration with CI/CD Pipelines

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

#### Local and CI Environment Consistency

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

#### Pre-Commit Hooks

Pre-commit hooks are scripts that run before a commit is made. They are a
powerful tool for ensuring that code quality checks are performed before
commits. This can include linting, formatting, and other checks such as testing.
This is a good to have as it injects some sort of discipline and automation into
the local development environment.

#### Order in Pipeline

In a CI/CD pipeline, the typical sequence is to lint first, then format, and
finally run unit tests. Linting first helps catch syntax errors and code smells
early, reducing the likelihood of these issues causing test failures. Formatting
next ensures code consistency, and finally, unit tests validate the
functionality. This order optimizes the build process, catching errors
efficiently and maintaining code quality.

## Formatting

What is formatting? Formatting is the process of ensuring that the codebase
adheres to a consistent style/format. This includes indentation, line length and
spacing, among other things. The goal is to make the codebase more readable and
maintainable. This will reduce friction in code reviews. Imagine the frustration
if developer A uses a 120 character line length and developer B uses 80
characters. They will not be in sync with each other.

### What is the Difference between Linting and Formatting?

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

### Tools

In the industry, there are a few leading tools:

-   [Black](https://black.readthedocs.io/en/stable/)
-   [Ruff](https://docs.astral.sh/ruff/)

Interestingly, `ruff` serves as both a linter and a formatter, so we can have an
all in one package. However, `black` seems to be the most popular formatter in
the Python and generally, more matured.

Many teams also add in `isort` to sort the imports. This is a good practice as
it makes the imports more readable.

### Best Practices for Formatting and CI/CD Pipeline Integration

In general, the best practices for formatting are similar to those for linting.
The key is to ensure that the formatting tool is integrated into the development
workflow consistently.

## Summary

In the realm of software development, styling, formatting, and linting serve as
foundational practices to ensure code quality, readability, and maintainability.
**Linting** plays a critical role in identifying potential errors and code
smells early in the development cycle, promoting higher code quality and
reducing technical debt. On the other hand, **formatting** focuses on the
aesthetic aspects of the code, adhering to conventions that make the codebase
uniformly readable and easier to maintain.

Integrating these practices into a **CI/CD pipeline** enhances the automation
cycle, allowing for consistent code quality checks with each commit or build.
Linters check for syntax errors and potential bugs, while formatters ensure the
code adheres to style guidelines, both acting before unit tests in the pipeline
to catch and correct issues early, streamlining development, and deployment
processes. This integration is pivotal in maintaining a high standard of code
quality throughout the software development lifecycle.

Frequent code commits and daily integration are crucial practices for reducing
conflicts and fostering collaboration within development teams. Regularly
committing code minimizes the risk of large-scale conflicts and promotes early
detection and resolution of minor issues. This approach enhances transparency,
facilitates progress tracking, and encourages team communication, ultimately
leading to a more cohesive and productive development process[^1].

## References and Further Readings

-   [Code Style Checks - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/continuous-integration/#code-style-checks)
-   [Code Analysis Linting - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/code-reviews/recipes/python/#code-analysis-linting)
-   [Differences between code linters and formatters](https://taiyr.me/what-is-the-difference-between-code-linters-and-formatters)
-   [Format Code vs Lint Code](https://medium.com/@awesomecode/format-code-vs-and-lint-code-95613798dcb3)
-   [PEP8 guide](https://peps.python.org/pep-0008/)
-   [Pre-commits Styling](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)

[^1]:
    [Deliver Quickly and Daily - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/continuous-integration/#deliver-quickly-and-daily)
