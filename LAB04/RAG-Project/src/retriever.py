


"""
include both steps "convert query to vector" + "search in FAISS" into a single class 
to make it easy for main.py and lab07 to use with a single function call.
"""

import config
from src.embedding_model import EmbeddingModel
from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore, load_chunk_store


class Retriever:
    """
    Retriever class combining FAISS dense vector search and BM25 keyword search.
    """
    def __init__(self, model_name=None, index_path=None, chunk_store_path=None):
        self.use_hybrid = config.USE_HYBRID
        try:
            self.hybrid_retriever = HybridRetriever()
            self.chunks = self.hybrid_retriever.chunks
        except Exception as e:
            print(f"[Retriever Warning] Could not initialize HybridRetriever: {e}")
            self.use_hybrid = False
            model_name = model_name or config.EMBEDDING_MODEL_NAME
            index_path = index_path or config.FAISS_INDEX_FILE
            chunk_store_path = chunk_store_path or config.CHUNK_STORE_FILE
            
            self.embedding_model = EmbeddingModel(model_name)
            self.vector_store = VectorStore()
            self.vector_store.load(index_path)
            self.chunks = load_chunk_store(chunk_store_path)

    def retrieve(self, query, top_k=3):
        """
        Receive a user query and return the top_k most relevant chunks using BM25 + Dense RRF
        (or Dense only if BM25 is disabled).
        """
        if self.use_hybrid and hasattr(self, 'hybrid_retriever'):
            return self.hybrid_retriever.retrieve(query, top_k=top_k)

        query_vector = self.embedding_model.encode_query(query)
        scores, indices = self.vector_store.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["dense_score"] = float(score)
            chunk["bm25_score"] = None
            results.append(chunk)

        return results
