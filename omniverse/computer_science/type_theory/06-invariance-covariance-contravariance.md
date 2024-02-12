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

# Invariance, Covariance and Contravariance

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from typing_extensions import reveal_type
from rich.pretty import pprint
```

## The Motivation

This section is not particulary easy. So let's open with an example, a classic
one written by the author of Python, Guido van Rossum.

```{code-cell} ipython3
def append_pi(lst: List[float]) -> None:
    lst += [3.14]

my_list = [1, 3, 5]  # type: List[int]

_ = append_pi(my_list)   # Naively, this should be safe..
pprint(my_list)
```

The function `append_pi` is supposed to append the value of $\pi$ to the list
`lst`. However, the type of `lst` is `List[float]`, and the type of `my_list` is
`List[int]`. The function `append_pi` is supposed to be safe, but it is not. Why
not? Isn't `int` (integers $\mathbb{Z}$) a subtype of `float` (real numbers
$\mathbb{R}$)? Did we not establish this in the section on subsumption?

Yes, we did. However, `int` being a subtype of `float` does not imply that
`List[int]` is a subtype of `List[float]`. Upon reflection, this makes sense,
because the above code would break the second criterion of subtyping (
{prf:ref}`type-theory-subtype-criterion`). The second criterion states that the
if the type $S$ is a subtype of $T$, then if $T$ has $N$ methods, then $S$ must
have at least the same set of $N$ methods. Consider a contradiction that
`List[int]` is a subtype of `List[float]`,

And indeed running through this piece of code via a static type checker like
`mypy` will raise an error.

```bash
6: error: Argument 1 to "append_pi" has incompatible type "list[int]"; expected "list[float]"  [arg-type]
    append_pi(my_list)   # Naively, this should be safe..
              ^~~~~~~
6: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
6: note: Consider using "Sequence" instead, which is covariant
```

The `mypy` error message even gave you some suggestion that `List` is
**invariant**.

## References and Further Readings

-   https://peps.python.org/pep-0483/#covariance-and-contravariance
-   https://peps.python.org/pep-0484/#covariance-and-contravariance
-   https://www.playfulpython.com/type-hinting-covariance-contra-variance/
-   https://nus-cs2030s.github.io/2021-s2/18-variance.html
-   https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generics
