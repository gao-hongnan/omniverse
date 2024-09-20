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
