import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from src.retriever import Retriever

# Configure Unicode support for console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

# Global retriever instance
retriever = None

def init_retriever():
    global retriever
    if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.CHUNK_STORE_FILE):
        print("[RAG-Web] Loading FAISS index and Chunk store...")
        retriever = Retriever(
            model_name=config.EMBEDDING_MODEL_NAME,
            index_path=config.FAISS_INDEX_FILE,
            chunk_store_path=config.CHUNK_STORE_FILE,
        )
        print(f"[RAG-Web] System ready! Total Chunks: {len(retriever.chunks)}")
    else:
        print("[RAG-Web] Warning: Vector DB files not found. Please build vector DB first.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    if retriever is None:
        return jsonify({
            "status": "error",
            "ready": False,
            "message": "Vector database not initialized"
        }), 500
    return jsonify({
        "status": "ok",
        "ready": True,
        "model": config.EMBEDDING_MODEL_NAME,
        "total_chunks": len(retriever.chunks)
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    if retriever is None:
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
        results = retriever.retrieve(query, top_k=top_k)
        
        # Clean and format response objects
        formatted_results = []
        for rank, item in enumerate(results, start=1):
            formatted_results.append({
                "rank": rank,
                "score": round(item.get("score", 0.0), 4),
                "answer": item.get("answer", item.get("text", "")),
                "question": item.get("question", ""),
                "category": item.get("category", "ทั่วไป")
            })

        return jsonify({
            "status": "success",
            "query": query,
            "top_k": top_k,
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
