# LAB05 — RAG System Development II

This lab follows the format of [DL-05-RAG System Development II](https://github.com/aproot-en/Advanced-Topic-in-Computer-Software-Course/tree/main/DL-05-RAG%20System%20Development%20II), which demonstrates 9 common LLM/RAG problems on one shared dataset.

The difference is the source of the problems. DL-05 *simulates* textbook failure modes. This lab documents the **10 problems that actually exist in my own LAB04 project** (`LAB04/RAG-Project`), and every simulation reads that project's real config, real source code and real chunk store — 60,499 chunks built from 4 MB of automotive data. Nothing here is invented: each number printed by a problem module is measured at run time.

## Structure

```text
LAB05/
├── README.md
├── data_loader.py                  # Shared loader: LAB04 config / source / chunks -> Python objects
├── main.py                         # Main menu for running each problem
├── problem01_hallucination.py      # Relevance threshold that can never fire
├── problem02_vocab_mismatch.py     # Query transform built for the wrong domain and language
├── problem03_data_quality.py       # 99.2% of the KB produced by a silent fallback parser
├── problem04_chunking.py           # CHUNK_SIZE / CHUNK_OVERLAP applied to 0.02% of the data
├── problem05_metadata.py           # Metadata that is constant, so filtering is impossible
├── problem06_reranking.py          # Reranker that always fails to load, silently
├── problem07_generation.py         # Dead branch, extra API call per question, language mismatch
├── problem08_config.py             # API key read from a placeholder env var -> LLM never runs
├── problem09_evaluation.py         # Evaluation suite that raises before writing one test item
└── problem10_index_staleness.py    # Dead staleness guard + 370 MB of build artifacts in git
```

## Dataset

There is no new dataset. The subject under test is `LAB04/RAG-Project` itself:

| Artifact | What it is | Size |
|---|---|---|
| `data/car_fundamentals_glossary_kb.txt` | Automotive glossary, section/topic format | 30 KB |
| `data/eu_and_jp_cars_dataset.txt` | 421 EU/JP vehicles, pipe-delimited | 68 KB |
| `data/CARS_DATASET_RAG.txt` | 6,000 vehicle records, `Record N:` blocks | 4 MB |
| `outputs/chunks.json` | The chunk store the project actually built | 60,499 chunks |
| `config.py` | All feature switches and thresholds | — |

`data_loader.py` imports the real `config.py` and reads the real `chunks.json`, so if the project is rebuilt with different settings, the numbers in this lab change with it.

## How to run

```bash
cd LAB05
python main.py          # interactive menu
python main.py 3        # run one problem
python main.py 0        # run all ten
```

Standard library only — no `faiss`, `sentence-transformers` or `openai` needed. The problems are demonstrated by reading the project's configuration, source code and chunk store, not by loading its models.

## Summary

| # | Problem | Main Idea | Evidence measured at run time |
|---|---------|-----------|-------------------------------|
| 1 | Hallucination | The "no relevant data" guard is unreachable. | `RELEVANCE_SCORE_THRESHOLD = 0.005`, but the lowest RRF score possible is `1/(60+20) = 0.0125`. Out-of-scope questions still get 3 chunks. |
| 2 | Vocabulary Mismatch | The query-transform layer belongs to the previous project. | `SLANG_MAP` holds 9 Thai sexual-health terms; they occur **0** times in the automotive KB. `REWRITE_PROMPT` still says "sexual health database". |
| 3 | Data Quality | The largest data file matches no parser. | 60,011 of 60,499 chunks (99.2%) come from the generic line loader and share one question. File headers were ingested as knowledge. |
| 4 | Chunking | Chunk size is decided by the loader, not by `text_splitter`. | Median chunk = 135 chars; only **12** chunks were long enough for `CHUNK_SIZE=600` to split. One vehicle record spans 7 separate chunks. |
| 5 | Metadata Filtering | Metadata is constant where it matters. | 99.2% of chunks have `category="General Knowledge"`. Filtering by brand returns 0 hits on the 6,000-vehicle dataset. |
| 6 | Re-ranking | The reranker cannot load and the failure is swallowed. | `config.RERANK_MODEL_NAME` does not exist -> `AttributeError` -> `get_reranker()` returns `None`, which the pipeline reads as "off". |
| 7 | Faithfulness | Retrieval can be right while generation is not. | `if is_connected: ... else: ...` runs identical code. `check_connection()` fires an extra API call per question. Thai prompts over an English KB. |
| 8 | RAG Configuration | One placeholder disables the whole LLM. | `os.getenv("[ENCRYPTION_KEY]")` returns `None` -> `None.strip()` -> `AttributeError` -> silent fallback to `NoLLM`, while `USE_LLM = True`. |
| 9 | Evaluation | The evaluation suite cannot produce a single metric. | `config.GOLDEN_SET_SIZE` is missing; `golden_set.json` was never created; the golden set's "slang" variant is Thai. |
| 10 | Index & Repo Hygiene | Nothing verifies that the index matches the data. | `warn_if_stale()` is never called and raises `KeyError: 'file'`. No `.gitignore`; two 177 MB artifacts are tracked, above GitHub's 100 MB limit. |

## What connects them

Problems 2, 7 and 9 share one root cause: LAB04 was forked from an earlier Thai sexual-health RAG project, the knowledge base was swapped for English automotive data, and the retrieval-side vocabulary, the prompt templates and the evaluation suite were never migrated. Traces are still visible in the code — `text_splitter.py` documents "in this answer in sex_q_a.txt", `rag_pipeline.py` uses "What should I do if a condom breaks?" as its usage example, and `run_web.py` calls itself the "Pad Krapao RAG System".

Problems 3, 4, 5 and 9 share a second root cause: `document_loader.load_qa_file()` dispatches on file name and falls back to a line-by-line parser for anything it does not recognise. Because that fallback never raises, the largest data file was shredded into 60,011 single-line chunks and every downstream stage — chunking, metadata, retrieval scoring, evaluation — inherited the damage.

Problems 1, 6, 8 and 10 share a third: failures are caught and turned into defaults. A missing config key, an unreachable threshold and a swallowed `AttributeError` all present as "the feature is switched off", so `show_settings()` describes the pipeline that was intended rather than the one that runs.

## References

- Course reference project: [aproot-en/Advanced-Topic-in-Computer-Software-Course — DL-05](https://github.com/aproot-en/Advanced-Topic-in-Computer-Software-Course/tree/main/DL-05-RAG%20System%20Development%20II) — problem-based simulation format, module layout and `main.py` menu structure are adapted from it.
- System under test: `LAB04/RAG-Project` in this repository.
