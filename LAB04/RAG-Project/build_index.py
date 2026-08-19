"""
Build all search indexes (FAISS Dense Vector + BM25 Keyword).
Run this script whenever the dataset or chunk settings are updated.

Usage:
    python build_index.py
"""

import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.embedding_model import EmbeddingModel
from src.hybrid_retriever import build_bm25, save_bm25
from src.vector_store import VectorStore, save_chunk_store
import src.index_meta as index_meta


def main():
    print("==========================================================================")
    print(" 🛠️  BUILDING ALL INDEXES (FAISS Dense Vector + BM25 Keyword)")
    print("==========================================================================")

    # 1. Check chunks file
    if not os.path.exists(config.CHUNKS_FILE):
        print(f"[build_index] Error: {config.CHUNKS_FILE} not found.")
        print("Please run lab01_extract_text.py and lab02_chunking.py first.")
        return

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[build_index] Loaded {len(chunks)} chunks from {config.CHUNKS_FILE}")

    # 2. Build or Load Embeddings
    if os.path.exists(config.EMBEDDINGS_FILE):
        print(f"[build_index] Loading existing embeddings from {config.EMBEDDINGS_FILE}")
        embeddings = np.load(config.EMBEDDINGS_FILE)
    else:
        print("[build_index] Generating embeddings with model:", config.EMBEDDING_MODEL_NAME)
        model = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
        texts = [c.get("text", "") for c in chunks]
        embeddings = model.encode(texts)
        np.save(config.EMBEDDINGS_FILE, embeddings)
        print(f"[build_index] Saved embeddings to {config.EMBEDDINGS_FILE}")

    # 3. Build & Save FAISS Index
    print("[build_index] Building FAISS Index...")
    store = VectorStore()
    store.build_index(embeddings)
    store.save(config.FAISS_INDEX_FILE)
    save_chunk_store(chunks, config.CHUNK_STORE_FILE)

    # 4. Build & Save BM25 Index
    print("[build_index] Building BM25 Index...")
    bm25 = build_bm25(chunks)
    save_bm25(bm25)

    # 5. Save index metadata
    try:
        index_meta.save(len(chunks))
        print(f"[build_index] Saved index metadata.")
    except Exception as e:
        print(f"[build_index] Index metadata notice: {e}")

    print("\n✅ All indexes successfully built and ready for RAG!")


if __name__ == "__main__":
    main()
