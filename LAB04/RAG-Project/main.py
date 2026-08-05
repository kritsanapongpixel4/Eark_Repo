"""
Main RAG Automotive Assistant.

Answer questions using the Automotive Knowledge Base (Retrieval-Based QA).
Before running, complete Lab 01 - Lab 04 to build the vector database.

run: python main.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.retriever import Retriever


def print_answer(rank, item):
    print(f"\nResult {rank} (Score: {item['score']:.4f})")
    print(f"Category: {item.get('category', 'Automotive Knowledge')}")
    print(f"Answer:\n{item['answer']}")
    print("-" * 70)


def main():
    if not os.path.exists(config.FAISS_INDEX_FILE):
        print("Error: Vector database not found.")
        print("Please run lab01_extract_text.py -> lab04_create_vector_db.py first.")
        return

    print("==========================================================================")
    print(" 🚗⚡ AUTOMOTIVE & CAR KNOWLEDGE RAG SYSTEM (2025-2026 Edition)")
    print("==========================================================================")
    print(" Ask any question about vehicle specs, body types, powertrains, or EV tech!")
    print(" Enter 'exit', 'quit', or 'q' to quit.\n")

    retriever = Retriever(
        model_name=config.EMBEDDING_MODEL_NAME,
        index_path=config.FAISS_INDEX_FILE,
        chunk_store_path=config.CHUNK_STORE_FILE,
    )

    while True:
        query = input("\n🚗 Ask a question about cars: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("\n=== Thank you for using Automotive RAG Assistant! Drive safely! 🏎️💨 ===")
            break

        if not query:
            continue

        results = retriever.retrieve(query, top_k=config.TOP_K)

        if not results:
            print("No relevant answer found in the automotive knowledge base.")
            continue

        for rank, item in enumerate(results, start=1):
            print_answer(rank, item)


if __name__ == "__main__":
    main()
