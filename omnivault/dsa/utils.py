from typing import Any, Sequence

from IPython.display import HTML, display

SUCCESS_STYLE = "color:green;"
ERROR_STYLE = "color:red;"


def compare_test_case(
    actual: Any,
    expected: Any,
    description: str = "",
) -> None:
    """Compare a single test case and display the result.

    Parameters
    ----------
    actual : Any
        The actual result to compare
    expected : Any
        The expected result
    description : str, optional
        Description of the test case, by default ""

    Returns
    -------
    None
        Displays HTML output with test results
    """
    try:
        assert actual == expected
        display(HTML(f'<span style="{SUCCESS_STYLE}">Test passed:</span> {description}'))
    except AssertionError:
        display(
            HTML(
                f'<span style="{ERROR_STYLE}">Test failed:</span> {description}<br>'
                f"Expected: {expected}, but got: {actual}"
            )
        )


def compare_test_cases(
    actual_list: Sequence[Any],
    expected_list: Sequence[Any],
    description_list: Sequence[str],
) -> None:
    """Compare multiple test cases and display results.

    Parameters
    ----------
    actual_list : Sequence[Any]
        Sequence of actual results
    expected_list : Sequence[Any]
        Sequence of expected results
    description_list : Sequence[str]
        Sequence of test descriptions

    Raises
    ------
    ValueError
        If input sequences have different lengths

    Returns
    -------
    None
        Displays HTML output with test results
    """
    if not (len(actual_list) == len(expected_list) == len(description_list)):
        raise ValueError(
            "All input sequences must have the same length. "
            f"Got lengths: actual={len(actual_list)}, "
            f"expected={len(expected_list)}, "
            f"descriptions={len(description_list)}"
        )

    for i, (actual, expected, description) in enumerate(
        zip(actual_list, expected_list, description_list, strict=False)
    ):
        compare_test_case(
            actual=actual,
            expected=expected,
            description=f"{description} - {i}",
        )
