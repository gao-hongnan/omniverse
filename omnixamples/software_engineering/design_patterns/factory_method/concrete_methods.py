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
