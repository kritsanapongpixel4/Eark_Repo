# -*- coding: utf-8 -*-
# Shared loader for problem01-10.
#
# Unlike DL-05 (which parses one flat .txt knowledge base), the "dataset" studied
# here is the LAB04 RAG project itself: its config, its source files, its data
# files and the chunk store it actually built. Every problem module reads real
# artifacts through this module instead of hard-coded sample data.
#
# Only the standard library is used, so the simulations run even on a machine
# with no faiss / sentence-transformers / openai installed.
import json
import os
import sys

# The knowledge base and the prompt templates mix English and Thai, so force the
# console to UTF-8 (same workaround the project uses in config.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "LAB04", "RAG-Project"))

CHUNKS_FILE = os.path.join(PROJECT_DIR, "outputs", "chunks.json")
CHUNK_STORE_FILE = os.path.join(PROJECT_DIR, "vector_db", "chunk_store.json")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

_chunks_cache = None
_config_cache = None


def project_exists():
    return os.path.isdir(PROJECT_DIR)


def load_config():
    # Import the real LAB04/RAG-Project/config.py, so every number printed by the
    # problem modules is the value the project is actually configured with.
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import config as project_config

    _config_cache = project_config
    return _config_cache


def load_chunks():
    # Returns the chunk list the project built (outputs/chunks.json).
    # Each chunk: chunk_id, qa_id, category, question, answer, text, part_idx, line_no
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache

    path = CHUNKS_FILE if os.path.exists(CHUNKS_FILE) else CHUNK_STORE_FILE
    if not os.path.exists(path):
        print(f"[data_loader] chunk file not found: {path}")
        print("[data_loader] Build it first with: python labs/lab02_chunking.py")
        _chunks_cache = []
        return _chunks_cache

    with open(path, encoding="utf-8") as f:
        _chunks_cache = json.load(f)
    return _chunks_cache


def generic_chunks(chunks=None):
    # Chunks produced by the fallback line-by-line loader in document_loader.py
    chunks = chunks if chunks is not None else load_chunks()
    return [c for c in chunks if c.get("category") == "General Knowledge"]


def parsed_chunks(chunks=None):
    # Chunks produced by a real, format-aware parser (glossary + EU/JP dataset)
    chunks = chunks if chunks is not None else load_chunks()
    return [c for c in chunks if c.get("category") != "General Knowledge"]


def read_source(relative_path):
    # Read a source file of the LAB04 project as text (for static inspection)
    path = os.path.join(PROJECT_DIR, relative_path.replace("/", os.sep))
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_data_lines(filename, limit=None):
    # Read raw lines of a knowledge-base file in LAB04/RAG-Project/data/
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        if limit is None:
            return f.read().splitlines()
        return [next(f).rstrip("\n") for _ in range(limit)]


def oneline(text, width=90):
    # Collapse a multi-line chunk into a single readable line for printing
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width] + "..."


def data_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(os.listdir(DATA_DIR))


if __name__ == "__main__":
    chunks = load_chunks()
    print("Project directory :", PROJECT_DIR)
    print("Total chunks      :", len(chunks))
    print("Generic-loader     :", len(generic_chunks(chunks)))
    print("Properly parsed    :", len(parsed_chunks(chunks)))
    print("Data files         :", data_files())
    if chunks:
        print("First chunk example:", {k: str(v)[:60] for k, v in chunks[0].items()})
