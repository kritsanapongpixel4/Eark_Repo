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
from src.rag_pipeline import RAGPipeline


def print_result(rag_result):
    print("\n" + "=" * 70)
    print("🤖 ANSWER SYNTHESIS (LM Generator):")
    print(rag_result.get("answer", ""))
    print("=" * 70)

    is_llm_connected = rag_result.get("llm_connected", False)
    llm_status = rag_result.get("llm_status", "")
    print(f"🔗 LM Connection Status: {'✅ Connected' if is_llm_connected else '⚠️ Offline / Fallback'} ({llm_status})")
    
    retrieved = rag_result.get("retrieved", [])
    print(f"🔍 RETRIEVED SOURCES (BM25 + FAISS Dense Hybrid - Top {len(retrieved)}):")
    for rank, item in enumerate(retrieved, start=1):
        bm25_str = f"{item['bm25_score']:.4f}" if item.get('bm25_score') is not None else "N/A"
        dense_str = f"{item['dense_score']:.4f}" if item.get('dense_score') is not None else "N/A"
        print(f"\n  [{rank}] RRF Score: {item.get('score', 0.0):.4f} | BM25: {bm25_str} | Dense: {dense_str}")
        print(f"      Q: {item.get('question', '')}")
        ans_preview = item.get('answer') or item.get('text', '')
        print(f"      A: {ans_preview[:120]}...")
    print("-" * 70)


def main():
    if not os.path.exists(config.FAISS_INDEX_FILE):
        print("Error: Vector database not found.")
        print("Please run lab01_extract_text.py -> lab04_create_vector_db.py first.")
        return

    print("==========================================================================")
    print(" 🚗⚡ CAR KNOWLEDGE RAG SYSTEM (BM25 Hybrid + LM Generator)")
    print("==========================================================================")
    print(" Ask any question about vehicle specs, body types, powertrains, or EV tech!")
    print(" Enter 'exit', 'quit', or 'q' to quit.\n")

    rag = RAGPipeline()
    rag.show_settings()

    while True:
        query = input("\n🚗 Ask a question about cars: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("\n=== Thank you for using Automotive RAG Assistant! Drive safely! 🏎️💨 ===")
            break

        if not query:
            continue

        rag_result = rag.ask(query, top_k=config.TOP_K)
        print_result(rag_result)


if __name__ == "__main__":
    main()
