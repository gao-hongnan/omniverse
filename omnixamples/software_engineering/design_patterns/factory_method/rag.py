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
        chunker: Chunker = ComponentFactory.get_component("chunker", chunker_type)
        embedder: Embedder = ComponentFactory.get_component("embedder", embedder_type)
        retriever: Retriever = ComponentFactory.get_component("retriever", retriever_type)
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
