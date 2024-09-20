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
