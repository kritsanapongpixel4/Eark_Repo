

"""
Project configuration.

Shared paths and constants used by all labs.
Avoid repeating hard-coded values.
"""

import os
import sys

# solve the problem of Windows console not showing Thai text (UnicodeEncodeError)
# configure stdout/stderr to use UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# main folder of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder structure:
# RAG-Project/
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Data file of the knowledge base
SOURCE_FILE = [os.path.join(DATA_DIR, "car_fundamentals_glossary_kb.txt"),
               os.path.join(DATA_DIR, "CARS_DATASET_RAG.txt"),
               os.path.join(DATA_DIR, "eu_and_jp_cars_dataset.txt")]

# results intermediate files (outputs/)
EXTRACTED_TEXT_FILE = os.path.join(OUTPUT_DIR, "extracted_text.json")
CHUNKS_FILE = os.path.join(OUTPUT_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
RETRIEVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "retrieval_results.json")

#file paths for vector database (vector_db/)
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "document.index")
CHUNK_STORE_FILE = os.path.join(VECTOR_DB_DIR, "chunk_store.json")
BM25_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "bm25_index.pkl")
INDEX_META_FILE = os.path.join(VECTOR_DB_DIR, "index_meta.json")

# settings for chunking and embedding
#data is already in Q&A format, but if the answer is too long, 
# it will be split into chunks of this size (number of characters)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# setting for the embedding model
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# RAG setting for the retrieval process
TOP_K = 3
CANDIDATE_K = 20
RRF_K = 60

# Hybrid Retrieval Settings (BM25 + Dense RRF)
USE_HYBRID = True
USE_RERANK = False

# Query Transformation Settings
USE_QUERY_TRANSFORM = False
QUERY_TRANSFORM_MODE = "rewrite"  # options: "rewrite", "multi_query", "hyde"
MULTI_QUERY_COUNT = 3

# Memory Settings
USE_MEMORY = True
MEMORY_MAX_TURNS = 5

# LLM / Generator Settings (ใช้บริการผ่าน Cloud LM API)
USE_LLM = True
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # ตัวเลือก: "gemini", "openai", "groq", "openrouter"
LLM_MODEL = os.getenv("LLM_MODEL", None)

# API Keys (ใส่ Key ตรงนี้ได้เลย หรือจะใส่ผ่าน Environment Variable ก็ได้)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Iuhg_Nd7Dp8iPFKrZjKtc7L1BsLaizrsNV3XLen6Qm3w")      # Google Gemini API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")      # ใส่ OpenAI API Key ที่นี่
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")          # ใส่ Groq API Key ที่นี่
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

LLM_PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-1.5-flash", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct", "OPENROUTER_API_KEY"),
}

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1000

NO_CONTEXT_MESSAGE = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในระบบฐานความรู้ยานยนต์"
DISCLAIMER = "⚠️ หมายเหตุ: ข้อมูลนี้จัดทำขึ้นเพื่อการให้ความรู้ด้านยานยนต์เท่านั้น"

# Evaluation settings
GOLDEN_SET_FILE = os.path.join(OUTPUT_DIR, "golden_set.json")
EVAL_RETRIEVAL_FILE = os.path.join(OUTPUT_DIR, "eval_retrieval_report.json")
EVAL_GENERATION_FILE = os.path.join(OUTPUT_DIR, "eval_generation_report.json")
EVAL_K_VALUES = [1, 3, 5]

# create output folders in advance if they don't exist
for _dir in (OUTPUT_DIR, VECTOR_DB_DIR):
    os.makedirs(_dir, exist_ok=True)
