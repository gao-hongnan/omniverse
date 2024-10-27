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

# Factory Method

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/tree/6ccd7496a4dd23853e0e2f806922e5d8343aaa8f/omnixamples/software_engineering/design_patterns/factory_method)

```{contents}
:local:
```

## Definition

The **Factory Method** is a **creational design pattern** that provides an
interface for creating objects in a superclass but allows subclasses to alter
the type of objects that will be created. Essentially, it delegates the
instantiation process to subclasses, promoting loose coupling and adherence to
the **Open/Closed Principle**—software entities should be open for extension but
closed for modification.

## Why "Factory Method"?

The term "Factory" signifies a component responsible for creating objects, much
like a manufacturing factory produces goods. The "Method" aspect refers to the
pattern's use of methods to encapsulate the creation logic. Therefore, the
**Factory Method** encapsulates object creation within methods, enabling
flexibility and scalability in how objects are instantiated and managed.

Here relate to the [named constructor](./named_constructor.md) is also loosely a
factory method.

## Factory Method in the RAG System

### RAG System Components

In the RAG system, several core components interact to process queries:

-   **Chunkers**: Divide input text into manageable chunks.
-   **Embedders**: Transform text chunks into numerical embeddings.
-   **Retrievers**: Fetch relevant documents based on embeddings.

Each of these components can have multiple implementations (e.g.,
`SimpleChunker`, `AdvancedChunker`), each with distinct behaviors and
performance characteristics.

### Implementing the Factory Method

In our implementation, the **Factory Method** pattern is embodied by the
`ComponentFactory` class. This factory is responsible for creating instances of
`Chunker`, `Embedder`, and `Retriever` based on specified identifiers.

````{tab} **abstract_methods**
```python
from abc import ABC, abstractmethod
from typing import List


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        ...


class Embedder(ABC):
    @abstractmethod
    def embed(self, chunks: List[str]) -> List[List[float]]:
        ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query_embedding: List[float]) -> List[str]:
        ...
```
````

````{tab} **concrete_methods**
```python
import hashlib
import re
from typing import List

from omnixamples.software_engineering.design_patterns.factory_method.abstract_methods import (
    Chunker,
    Embedder,
    Retriever,
)


class SimpleChunker(Chunker):
    def chunk(self, text: str) -> List[str]:
        return text.split(".")


class AdvancedChunker(Chunker):
    def chunk(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?]) +", text)


class SimpleEmbedder(Embedder):
    def embed(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for chunk in chunks:
            embedding = [float(ord(c)) / 1000.0 for c in chunk if c.isalpha()]
            embeddings.append(embedding[:10])  # Limit to first 10 values for simplicity
        return embeddings


class AdvancedEmbedder(Embedder):
    def embed(self, chunks: List[str]) -> List[List[float]]:
        # Placeholder for advanced embedding logic, e.g., using transformers
        return [[0.1 * i for i in range(10)] for _ in chunks]


class SimpleRetriever(Retriever):
    def retrieve(self, query_embedding: List[float]) -> List[str]:
        hash_val = int(hashlib.sha256(str(query_embedding).encode("utf-8")).hexdigest(), 16)
        return [f"Document_{hash_val % 100}" for _ in range(3)]


class AdvancedRetriever(Retriever):
    def retrieve(self, query_embedding: List[float]) -> List[str]:
        # Placeholder for advanced retrieval logic, e.g., semantic search
        hash_val = int(hashlib.sha256(str(query_embedding).encode("utf-8")).hexdigest(), 16)
        return [f"AdvancedDoc_{hash_val % 100}" for _ in range(3)]
```
````

````{tab} **registry**
```python
from __future__ import annotations

from abc import ABC
from typing import Callable, Dict, Type, TypeVar

from omnixamples.software_engineering.design_patterns.factory_method.abstract_methods import (
    Chunker,
    Embedder,
    Retriever,
)
from omnixamples.software_engineering.design_patterns.factory_method.concrete_methods import (
    AdvancedChunker,
    AdvancedEmbedder,
    AdvancedRetriever,
    SimpleChunker,
    SimpleEmbedder,
    SimpleRetriever,
)

C = TypeVar("C", bound=ABC)


class ComponentFactory:
    """Factory class for creating components.
    {
    'chunker': {
        'simple': <function create_simple_chunker at 0x102258670>,
        'advanced': <function create_advanced_chunker at 0x1023d53a0>
    },
    'embedder': {
        'simple': <function create_simple_embedder at 0x1023d5700>,
        'advanced': <function create_advanced_embedder at 0x1023d58b0>
    },
    'retriever': {'simple': <function create_simple_retriever at 0x1023d5b80>}
    }
    """

    _creators: Dict[str, Dict[str, Callable[[], C]]] = {}

    @classmethod
    def register(
        cls: Type[ComponentFactory], category: str, identifier: str
    ) -> Callable[[Callable[[], C]], Callable[[], C]]:
        def decorator(creator: Callable[[], C]) -> Callable[[], C]:
            if category not in cls._creators:
                cls._creators[category] = {}
            cls._creators[category][identifier] = creator
            return creator

        return decorator

    @classmethod
    def get_component(cls: Type[ComponentFactory], category: str, identifier: str) -> C:
        try:
            creator = cls._creators[category][identifier]
            return creator()
        except KeyError as exc:
            available = cls._creators.get(category, {})
            raise ValueError(
                f"Unknown {category} type: '{identifier}'. Available types: {list(available.keys())}"
            ) from exc


# Register Chunkers
@ComponentFactory.register(category="chunker", identifier="simple")
def create_simple_chunker() -> Chunker:
    return SimpleChunker()


@ComponentFactory.register(category="chunker", identifier="advanced")
def create_advanced_chunker() -> Chunker:
    return AdvancedChunker()


# Register Embedders
@ComponentFactory.register(category="embedder", identifier="simple")
def create_simple_embedder() -> Embedder:
    return SimpleEmbedder()


@ComponentFactory.register(category="embedder", identifier="advanced")
def create_advanced_embedder() -> Embedder:
    return AdvancedEmbedder()


# Register Retrievers
@ComponentFactory.register(category="retriever", identifier="simple")
def create_simple_retriever() -> Retriever:
    return SimpleRetriever()


@ComponentFactory.register(category="retriever", identifier="advanced")
def create_advanced_retriever() -> Retriever:
    return AdvancedRetriever()
```
````

````{tab} **rag**
```python
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from omnixamples.software_engineering.design_patterns.factory_method.abstract_methods import (
    Chunker,
    Embedder,
    Retriever,
)
from omnixamples.software_engineering.design_patterns.factory_method.registry import ComponentFactory


class RAGSystem(BaseModel):
    chunker: Chunker
    embedder: Embedder
    retriever: Retriever

    @staticmethod
    def create_system(chunker_type: str, embedder_type: str, retriever_type: str) -> RAGSystem:
        chunker = ComponentFactory.get_component("chunker", chunker_type)
        embedder = ComponentFactory.get_component("embedder", embedder_type)
        retriever = ComponentFactory.get_component("retriever", retriever_type)
        return RAGSystem(chunker=chunker, embedder=embedder, retriever=retriever)

    def process_query(self, query: str) -> List[str]:
        chunks = self.chunker.chunk(query)
        embeddings = self.embedder.embed(chunks)
        if not embeddings:
            return []
        query_embedding = embeddings[0]
        retrieved_docs = self.retriever.retrieve(query_embedding)
        return retrieved_docs

    class Config:
        arbitrary_types_allowed = True
```
````

````{tab} **client**
```python
from omnixamples.software_engineering.design_patterns.factory_method.rag import RAGSystem


def main() -> None:
    # Configuration for the RAG system
    configurations = [
        {"chunker_type": "simple", "embedder_type": "simple", "retriever_type": "simple"},
        {"chunker_type": "advanced", "embedder_type": "advanced", "retriever_type": "advanced"},
        {"chunker_type": "advanced", "embedder_type": "simple", "retriever_type": "advanced"},
    ]

    # Example query
    query = (
        "Implementing design patterns can greatly improve software architecture. "
        "Factory Method is one such pattern. "
        "It provides a way to delegate the instantiation logic to subclasses."
    )

    for idx, config in enumerate(configurations, start=1):
        print(f"\n--- RAG System Configuration {idx} ---")
        try:
            rag_system = RAGSystem.create_system(
                chunker_type=config["chunker_type"],
                embedder_type=config["embedder_type"],
                retriever_type=config["retriever_type"],
            )
            retrieved_documents = rag_system.process_query(query)
            print("Retrieved Documents:")
            for doc in retrieved_documents:
                print(f"- {doc}")
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
```
````

### What Are The Key Elements?

1. **Abstract Base Classes (ABCs)**: Define the interfaces (`Chunker`,
   `Embedder`, `Retriever`) that all concrete implementations must follow.

2. **Concrete Implementations**: Provide specific behaviors for each component
   (e.g., `SimpleChunker`, `AdvancedEmbedder`).

3. **ComponentFactory**: A generic factory that uses a registry to map component
   categories and identifiers to their respective creation methods.

4. **Registration Decorators**: Allow components to register themselves with the
   factory, specifying their category and identifier.

5. **RAGSystem Assembly**: Utilizes the factory to instantiate required
   components based on configuration.

### How It Works?

1. **Registration**: Each concrete component class registers itself with the
   `ComponentFactory` using a decorator, specifying its category (`chunker`,
   `embedder`, `retriever`) and a unique identifier (`simple`, `advanced`).

    ```python
    @ComponentFactory.register(category='chunker', identifier='simple')
    def create_simple_chunker() -> Chunker:
        return SimpleChunker()
    ```

2. **Instantiation**: When assembling the `RAGSystem`, the factory's
   `get_component` method is invoked with the desired category and identifier to
   obtain an instance of the required component.

    ```python
    rag_system = RAGSystem.create_system(
        chunker_type='simple',
        embedder_type='advanced',
        retriever_type='simple'
    )
    ```

3. **Usage**: The `RAGSystem` uses the instantiated components to process
   queries seamlessly, without needing to know the specifics of each component's
   implementation.

## Why Use the Factory Method Pattern?

### a. Encapsulation of Object Creation

By centralizing the instantiation logic within the factory, the system decouples
object creation from its usage. This means that the rest of the system doesn't
need to know the details of how components are created, fostering a separation
of concerns.

### b. Adherence to the Open/Closed Principle

The Factory Method allows the system to be **open for extension** but **closed
for modification**. New component types can be added without altering existing
factory logic or the `RAGSystem` assembly code. For example, introducing an
`AdvancedRetriever` only involves creating the class and registering it with the
factory.

### c. Promoting Loose Coupling

Components are not tightly bound to their implementations. The `RAGSystem`
interacts with components through their abstract interfaces (`Chunker`,
`Embedder`, `Retriever`), allowing for interchangeable implementations without
impacting the system's core logic.

### d. Enhancing Scalability and Maintainability

As the system grows, managing component instantiations becomes more
straightforward. Adding, removing, or modifying components doesn't require
widespread changes across the codebase. The factory serves as a single point of
management, simplifying maintenance and scalability.

### e. Facilitating Testing and Mocking

With a centralized factory, testing becomes more manageable. Mock components can
be registered with the factory during tests, allowing for controlled and
predictable testing environments without altering the system's core logic.

## We Adhere To Good Design Principles

### a. Single Responsibility Principle (SRP)

Each class has a single responsibility:

-   **Components (Chunkers, Embedders, Retrievers)**: Handle their specific
    processing logic.
-   **ComponentFactory**: Manages the creation of components.
-   **RAGSystem**: Orchestrates the interaction between components to process
    queries.

This segregation ensures that changes in one area (e.g., adding a new
`Embedder`) don't ripple across unrelated parts of the system.

### b. Dependency Inversion Principle (DIP)

High-level modules (`RAGSystem`) depend on abstractions (`Chunker`, `Embedder`,
`Retriever`), not on concrete implementations. The factory provides these
abstractions, allowing high-level modules to remain agnostic of low-level
details.

### c. Open/Closed Principle (OCP)

The system is designed to accommodate new components without modifying existing
code. This is achieved through the factory's registration mechanism, where new
components can be introduced by simply registering them, leaving existing code
untouched.

### d. Interface Segregation Principle (ISP)

Components adhere to their respective interfaces, ensuring that they only
implement the methods relevant to their roles. This prevents bloated interfaces
and promotes cleaner, more focused component designs.

### e. Liskov Substitution Principle (LSP)

Subtypes (e.g., `AdvancedChunker`) can replace their base types (`Chunker`)
without altering the correctness of the program. The factory ensures that
components instantiated via the factory conform to their abstract interfaces,
maintaining substitutability.

## References And Further Readings

-   https://refactoring.guru/design-patterns/factory-method/python/example
