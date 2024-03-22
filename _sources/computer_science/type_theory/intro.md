# Type Theory, A Very Rudimentary Introduction

In [mathematics](https://en.wikipedia.org/wiki/Mathematics) and
[theoretical computer science](https://en.wikipedia.org/wiki/Computer_science),
a **type theory** is the formal presentation of a specific
[type system](https://en.wikipedia.org/wiki/Type_system). The lineage of type
theory can be traced back to after the development of
[set theory](https://en.wikipedia.org/wiki/Set_theory) in the late 19th century
where it is born out of the need to avoid the
[Russell's paradox](https://en.wikipedia.org/wiki/Russell%27s_paradox).

In the context of computer science and programming, it is known that
[static program analysis](https://en.wikipedia.org/wiki/Static_program_analysis),
such as the type checking algorithms in the
[semantic analysis](<https://en.wikipedia.org/wiki/Semantic_analysis_(compilers)>)
phase of [compilers](https://en.wikipedia.org/wiki/Compiler), can be used to
detect type errors in compile time - and has deep connections to type theory.

In what follows, I will provide a very rudimentary introduction to type theory.
The series of post serve more as a reflection and learning experience for me
than an in-depth guide. We will walk through the basic concepts of type theory
and its applications in computer science. Most examples will be in
[Python](https://www.python.org/). It would be helpful if the readers skim
through [PEP 483 - The Theory of Type Hints](https://peps.python.org/pep-0483/),
written by [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum)
and [Ivan Levkivskyi](https://ie.linkedin.com/in/ivan-levkivskyi-186961125),
before diving into the series.

Although the reference guide is written by the authors of the python language,
the concepts of type theory, including subtype relationships and type safety,
are applicable across many programming languages, not just Python. These
principles have deep roots in computer science and are essential in
understanding static type checking in languages like Java, C#, TypeScript, and
many others.

## Table of Contents

```{tableofcontents}

```

## Citations

-   [1] Z. Luo, S. Soloviev, and T. Xue,
    ["Coercive subtyping: Theory and implementation"](https://www.sciencedirect.com/science/article/pii/S0890540112001757),
    Information and Computation, vol. 223, pp. 18–42, Feb. 2013.
    doi:10.1016/j.ic.2012.10.020
-   [2] C. Muñoz,
    ["Type Theory and Its Applications to Computer Science"](https://shemesh.larc.nasa.gov/fm/papers/ICASE1999-QNews.pdf),
    National Institute of Aerospace, Hampton, VA, Tech. Rep., Apr. 10, 2007.
