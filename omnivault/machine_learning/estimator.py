from __future__ import annotations


class BaseEstimator:
    """Base Class for Estimators."""

    def __repr__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        repr : str
            The string representation of the estimator.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        str : str
            The string representation of the estimator.
        """
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Check if two estimators are equal.

        Parameters
        ----------
        other : object
            The other estimator but can accept any object - this is the default
            behavior of the `__eq__` method because it allows comparison with any
            object.

        Returns
        -------
        eq : bool
            True if the estimators are equal, False otherwise.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Get the hash of the estimator.

        Returns
        -------
        hash : int
            The hash of the estimator.
        """
        return hash(tuple(sorted(self.__dict__.items())))
