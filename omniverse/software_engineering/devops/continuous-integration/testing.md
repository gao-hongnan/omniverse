---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Testing

```{contents}
:local:
```

First and foremost, this piece of article is not to teach you **_how_** to write
tests, because that would take a whole book to cover. Writing tests is "easy",
but writing **_good tests_** is an art, and difficult to master. I want to set
the stage for you to understand the importance of testing, and what types of
testing are there, along with some intuition.

In the fast-paced world of large language models (LLMs) and the applications
built on them, making sure these systems are reliable and sturdy is essential. A
solid testing strategy is key, and that's where the testing pyramid comes in.
This framework isn't just for traditional software; it's crucial for LLMs too,
helping ensure they work right across different scenarios.

## The Testing Pyramid

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

-   https://martinfowler.com/articles/practical-test-pyramid.html

-   https://microsoft.github.io/code-with-engineering-playbook/automated-testing/e2e-testing/testing-comparison/

## Unit Testing

Unit testing is a fundamental tool in every developer's toolbox. Unit tests not
only help us test our code, they encourage good design practices, reduce the
chances of bugs reaching production, and can even serve as examples or
documentation on how code functions. Properly written unit tests can also
improve developer efficiency.

### Intuition

Unit tests are the smallest and most granular tests in the testing pyramid. This
can be explained through an analogy of a building. If you think of your
application as a building, unit tests are the bricks. They are the smallest,
most fundamental building blocks of your application. They test the smallest
pieces of code, such as functions, methods, or classes, in isolation from the
rest of the application.

You need to ensure each brick is solid and reliable before you can build a
sturdy, reliable building. Similarly, you need to ensure each unit of code is
solid and reliable before you can build a sturdy, reliable application.

### Benefits of Unit Testing

#### Early Bug Detection and Reduce Cost

Why can't I catch a bug when the application is in production? Cost. It is
expensive to revert back and fix the bug. It is much cheaper to fix the bug when
it is caught early in the development cycle. Unit tests allow for the detection
of problems early in the development cycle, saving time and effort by preventing
bugs from propagating to later stages.

A
[2008 research study by IBM](https://www.researchgate.net/figure/IBM-System-Science-Institute-Relative-Cost-of-Fixing-Defects_fig1_255965523)
estimates that a bug caught in production could cost 6 times as much as if it
was caught during implementation[^1].

#### Refactoring with Confidence

Development is an **_iterative process_**. You write code, test it, and then
refactor it. You repeat this process until you are satisfied with the result.

With a suite of unit tests, developers can make changes to the codebase
confidently, knowing that they'll be alerted if a change inadvertently breaks
something that used to work.

#### Unit Test As Documentation

Unit tests serve as a form of documentation that describes what the code is
supposed to do, helping new developers understand the project's functionality
more quickly.

### Dependency Injection

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

#### Link to Unit Testing

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

#### No Dependency Injection vs Dependency Injection

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

### Stubs and Mocks

In unit testing, **mocks** and **stubs** are both types of test doubles used to
simulate the behavior of real objects in a controlled way. They are essential
tools for isolating the piece of code under test, ensuring that tests are fast,
reliable, and independent of external factors or system states. However, mocks
and stubs serve slightly different purposes and are used in different scenarios.

#### Stubs

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

#### Mocks

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

### Further Readings

-   [Unit Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/unit-testing/)
-   [Unit vs Integration vs System vs E2E Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/e2e-testing/testing-comparison/)

## Integration Testing

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
together as intended[^2].

### Intuition

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

### Benefits of Integration Testing

#### Exposes Interface Issues

Integration testing is crucial for detecting problems that occur when different
parts of a system interact. It can uncover issues with the interfaces between
components, such as incorrect data being passed between modules, or problems
with the way components use each other's APIs.

#### Validates Functional Coherence

By testing a group of components together, integration testing ensures that the
software functions correctly as a whole. This is particularly important for
critical paths in an application where the interaction between components is
complex or involves external systems like databases or third-party services.

#### Highlights Dependency Problems

Complex systems often rely on external dependencies, and integration testing can
reveal issues with these dependencies that might not be apparent during unit
testing. This includes problems with network communications, database
integrations, and interactions with external APIs.

#### Improves Confidence in System Stability

Successful integration tests provide confidence that the system will perform as
expected under real-world conditions. This is especially important when changes
are made to one part of the system, as it helps ensure that such changes do not
adversely affect other parts.

Given the provided overview of integration testing, let's construct a clear and
practical guide to implementing integration testing, focusing on a hypothetical
banking application as mentioned. This guide will outline key steps,
considerations, and an example to illustrate how integration testing can be
effectively applied.

### Understanding Integration Testing in Practice

Let's adopt the example given in
[Microsoft's Code with Engineering Playbook Integration Testing Design Blocks](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/#integration-testing-design-blocks)
to understand how integration testing can be applied in practice.

**Objective**: To ensure that independently developed modules of a banking
application—login, transfers, and current balance—work together as intended.

#### Step 1: Identify Integration Points

First, identify the key integration points within the application that require
testing. For the banking application, these points include:

-   **Login to Current Balance**: After a successful login, the application
    redirects the user to their current balance page with the correct balance
    displayed.
-   **Transfers to Current Balance**: After a transfer is initiated, ensure that
    the transfer completes successfully and the current balance is updated
    accurately.

#### Step 2: Design Integration Tests

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

### Further Readings

-   [Integration Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/)

## Unit vs Integration vs System vs E2E Testing

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

[^1]:
    [Why Unit Tests - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/unit-testing/why-unit-tests/)

[^2]:
    [Why Integration Testing - Microsoft](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/integration-testing/#why-integration-testing)
