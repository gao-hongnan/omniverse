# Type Theory, A Very Rudimentary Introduction

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Structured_Musings-purple)

In [mathematics](https://en.wikipedia.org/wiki/Mathematics) and
[theoretical computer science](https://en.wikipedia.org/wiki/Computer_science),
a **type theory** is the formal presentation of a specific
[type system](https://en.wikipedia.org/wiki/Type_system). The lineage of type
theory can be traced back to after the development of
[set theory](https://en.wikipedia.org/wiki/Set_theory) in the late 19th century
where it is born out of the need to avoid the
[Russell's paradox](https://en.wikipedia.org/wiki/Russell%27s_paradox). Muñoz
[2] gives a gentle survey of this history and of type theory's applications in
computer science.

In the context of computer science and programming, it is known that
[static program analysis](https://en.wikipedia.org/wiki/Static_program_analysis),
such as the type checking algorithms in the
[semantic analysis](<https://en.wikipedia.org/wiki/Semantic_analysis_(compilers)>)
phase of [compilers](https://en.wikipedia.org/wiki/Compiler), can be used to
detect type errors at static-analysis time - and has deep connections to type
theory.

In what follows, I will provide a very rudimentary introduction to type theory.
The series of posts serve more as a reflection and learning experience for me
than an in-depth guide. We will walk through the basic concepts of type theory
and its applications in computer science. Most examples will be in
[Python](https://www.python.org/), but the concepts — subtype relationships,
type safety, variance — apply across statically checked languages like Java,
C#, and TypeScript.

## How to read this series

The canonical reference for Python's type system today is the maintained
[typing specification](https://typing.python.org/en/latest/spec/). The
foundational PEPs —
[PEP 483, The Theory of Type Hints](https://peps.python.org/pep-0483/) and
[PEP 484, Type Hints](https://peps.python.org/pep-0484/) by Guido van Rossum
and Ivan Levkivskyi — remain worth reading as historical design records, but
where they and the specification disagree, the specification wins.

Conventions used throughout the series:

-   Code examples target **Python 3.14** and use modern syntax
    ([PEP 695](https://peps.python.org/pep-0695/) type parameters, builtin
    generics, `X | Y` unions). Legacy forms (`TypeVar(...)`, `Generic[T]`,
    capitalized `typing` aliases) appear only in sidebars explicitly labeled as
    legacy.
-   Every static-analysis claim is verified against **pyright** and **mypy**;
    where the two checkers disagree, both outputs are shown.
-   Runtime-demonstrable behavior runs in executed code cells; checker-rejected
    code lives in static blocks with the checker's actual output.

## Roadmap

1. {doc}`Subtypes <01-subtypes>` — types as sets of values; nominal versus
   structural subtyping; where structural subtyping meets the Liskov
   substitution principle.
2. {doc}`Type Safety <02-type-safety>` — safe substitution; why `int` is
   promoted to, not a subtype of, `float`; static, dynamic, and gradual
   typing.
3. {doc}`Subsumption <03-subsumption>` — the formal three-part criterion for
   subtypehood; reflexivity and transitivity; narrowing values and widening
   functions.
4. {doc}`Generics <04-generics>` — motivation for parameterized types; type
   variables, parameters, and arguments; generic classes, functions, and
   methods.
5. PEP 695: The Type Parameter Syntax *(upcoming)* — declaration sites,
   scoping rules, the `type` alias statement, lazy evaluation, and when legacy
   `TypeVar` survives.
6. {doc}`Bounds and Constraints <05-typevar-bound-constraints>` — restricting
   type variables by upper bounds and constraint sets (and, soon, defaults per
   PEP 696).
7. {doc}`Invariance, Covariance, Contravariance <06-invariance-covariance-contravariance>`
   — variance of type constructors; mutability's role; contravariance of
   `Callable` arguments.
8. Type Narrowing: `TypeIs` and `TypeGuard` *(upcoming)* — user-defined
   predicates, and why `TypeIs` soundness is a subtyping condition.
9. {doc}`Function Overloading <07-overload>` — `@overload` variants, runtime
   behavior, single dispatch, and unsafe overlapping overloads.
10. TypedDict and `ReadOnly` *(upcoming)* — structural typing of dict-shaped
    data; read-only keys as the payoff of the variance chapter.
11. {doc}`Sentinel Values <08-pep-661-sentinel-values>` — `NotGiven`/`Missing`
    singletons, why hand-rolled sentinels defeat narrowing, and the modern
    idioms.
12. Annotations at Runtime *(upcoming)* — PEP 649/749 deferred evaluation,
    `annotationlib`, and the end of `from __future__ import annotations`.

## Table of Contents

```{tableofcontents}

```

## Citations and Further Reading

-   [1] Z. Luo, S. Soloviev, and T. Xue,
    ["Coercive subtyping: Theory and implementation"](https://www.sciencedirect.com/science/article/pii/S0890540112001757),
    Information and Computation, vol. 223, pp. 18–42, Feb. 2013.
    doi:10.1016/j.ic.2012.10.020 — formal treatment of the coercive
    implementation of subtyping discussed at the end of the
    {doc}`subtypes chapter <01-subtypes>`.
-   [2] C. Muñoz,
    ["Type Theory and Its Applications to Computer Science"](https://shemesh.larc.nasa.gov/fm/papers/ICASE1999-QNews.pdf),
    National Institute of Aerospace, Hampton, VA, Tech. Rep., Apr. 10, 2007.
-   [Python typing specification](https://typing.python.org/en/latest/spec/) —
    the living, canonical description of the type system the checkers
    implement.
