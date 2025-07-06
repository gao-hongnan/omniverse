---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Context Object Pattern (God Object)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

Although some say it's an anti-pattern if used improperly, I think if used
properly, it can be a very powerful pattern as it empowers data sharing and
encapsulation. The tagline _God Object To Rule Them All_ is indeed true and can
be difficult to maintain - but if you compose them well, then each sub-component
can be decoupled and each sub-component can be tested and maintained
independently.

Especially in machine learning and deep learning, it's very common to have a lot
of parameters and configurations that are needed for the system to run, and it's
very easy to end up with a billion of hyperparameters and configurations. People
would then try to group all these parameters into a single object, which is the
god object.

See
[the series of articles on config management](../config_management/concept.md)
and you will see how I composed the configurations into a single object.

1. Link to state and config and metadata management.
2. Closely related to:
    1. Dependency Injection
    2. Service Locator
    3. Context Object
    4. Parameter Object
3. Some will say god object if it stores too many things
4. So you can compose them!
5. Link to dataclass and data class code smell.

-   https://stackoverflow.com/questions/771983/what-is-context-object-design-pattern
-   https://stackoverflow.com/questions/986865/can-you-explain-the-context-design-pattern/986947#986947
-   http://www.corej2eepatterns.com/ContextObject.htm
-   https://refactoring.guru/introduce-parameter-object
