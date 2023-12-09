from typing import Any, Callable, List

from IPython.core.display import HTML, display_html


# mypy: disable-error-code="no-untyped-call"
class TestFramework:
    """
    A simple testing framework for executing tests with descriptions.

    Attributes
    ----------
    tests: (List[Callable])
        A list to store the test functions.
    """

    def __init__(self) -> None:
        """Initialize the TestFramework object."""
        self.tests: List[Callable[[], None]] = []

    def describe(self, description: str) -> Callable[[Callable[[], None]], None]:
        """
        Decorator for describing a test.

        Parameters
        ----------
        description: str
            The description of the test.

        Returns
        -------
        wrapper: Callable
            A wrapped function with a description.
        """

        def wrapper(func: Callable[[], None]) -> None:
            """Execute the function and display its description."""
            display_html(HTML(f"<span style='color: blue;'>Description: {description}</span>"))
            func()

        return wrapper

    def individual_test(self, description: str) -> Callable[[Callable[[], None]], None]:
        """
        Decorator for running a single test.

        Parameters
        ----------
        description: str
            Description of what the test should do.

        Returns
        -------
        wrapper: Callable
            A wrapped function that runs the test.
        """

        def wrapper(func: Callable[[], None]) -> None:
            """Execute the function and display the test result."""
            try:
                func()
                display_html(HTML(f"<span style='color: green;'>  [Pass]</span> {description}"))
            except AssertionError as e:
                display_html(HTML(f"<span style='color: red;'>  [Fail]</span> {description} - {e}"))

        return wrapper

    def assert_equals(self, actual: Any, expected: Any, message: str) -> None:
        """
        Assert that the actual and expected values are equal.

        Parameters
        ----------
        actual  : Any
            The actual value.
        expected: Any
            The expected value.
        message : str
            The message to display if the assertion fails.

        Raises
        ------
        AssertionError: If the actual and expected values are not equal.
        """
        try:
            assert actual == expected, message
        except AssertionError as err:
            message = f"Test failed: {message}\nExpected: {expected}, but got: {actual}"
            raise AssertionError(message) from err
