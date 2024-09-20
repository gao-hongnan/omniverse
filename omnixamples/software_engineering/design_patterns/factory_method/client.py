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
