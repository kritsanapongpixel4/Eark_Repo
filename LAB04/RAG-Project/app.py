import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from src.rag_pipeline import RAGPipeline

# Configure Unicode support for console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

# Global RAG Pipeline instance
rag_pipeline = None

def init_retriever():
    global rag_pipeline
    if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.CHUNK_STORE_FILE):
        print("[RAG-Web] Initializing RAG Pipeline (BM25 + FAISS + LM Generator)...")
        rag_pipeline = RAGPipeline()
        print(f"[RAG-Web] System ready! BM25 Hybrid: {config.USE_HYBRID} | Total Chunks: {len(rag_pipeline.retriever.chunks)}")
    else:
        print("[RAG-Web] Warning: Vector DB files not found. Please build vector DB first.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    if rag_pipeline is None:
        return jsonify({
            "status": "error",
            "ready": False,
            "message": "RAG pipeline not initialized"
        }), 500

    is_llm_ok, llm_msg = rag_pipeline.generator.llm.check_connection()

    return jsonify({
        "status": "ok",
        "ready": True,
        "model": config.EMBEDDING_MODEL_NAME,
        "total_chunks": len(rag_pipeline.retriever.chunks),
        "bm25_enabled": config.USE_HYBRID,
        "bm25_index_exists": os.path.exists(config.BM25_INDEX_FILE),
        "llm_connected": is_llm_ok,
        "llm_status": llm_msg,
        "llm_provider": getattr(rag_pipeline.generator.llm, "provider", "none"),
        "llm_model": getattr(rag_pipeline.generator.llm, "model", "none"),
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    if rag_pipeline is None:
        return jsonify({
            "status": "error",
            "message": "Retrieval system is not initialized. Please ensure vector DB exists."
        }), 500

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", config.TOP_K))

    if not query:
        return jsonify({
            "status": "error",
            "message": "Empty query provided"
        }), 400

    try:
        rag_result = rag_pipeline.ask(query, top_k=top_k)
        retrieved_chunks = rag_result.get("retrieved", [])
        
        # Clean and format response objects for UI
        formatted_results = []
        for rank, item in enumerate(retrieved_chunks, start=1):
            formatted_results.append({
                "rank": rank,
                "score": round(float(item.get("score", 0.0)), 4),
                "bm25_score": round(float(item.get("bm25_score", 0.0)), 4) if item.get("bm25_score") is not None else None,
                "dense_score": round(float(item.get("dense_score", 0.0)), 4) if item.get("dense_score") is not None else None,
                "answer": item.get("answer", item.get("text", "")),
                "question": item.get("question", ""),
                "category": item.get("category", "Automotive Knowledge")
            })

        return jsonify({
            "status": "success",
            "query": query,
            "top_k": top_k,
            "answer": rag_result.get("answer", ""),
            "llm_connected": rag_result.get("llm_connected", False),
            "llm_status": rag_result.get("llm_status", ""),
            "timings": rag_result.get("timings", {}),
            "results": formatted_results
        })
    except Exception as e:
        print(f"[RAG-Web Error] {e}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    init_retriever()
    print("[RAG-Web] Starting Flask server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
