# -*- coding: utf-8 -*-
# Problem 10: Index staleness and repository hygiene — the project has a guard
# against serving results from an outdated index, but it is never called, and it
# crashes if it ever is. Meanwhile every index artifact is committed to git.
import json
import os
import subprocess

from data_loader import PROJECT_DIR, load_config, read_source


def repo_root():
    return os.path.normpath(os.path.join(PROJECT_DIR, "..", ".."))


def tracked_sizes():
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root(), capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    files = [p for p in out.stdout.split("\0") if p]
    sized = []
    for rel in files:
        full = os.path.join(repo_root(), rel)
        if os.path.isfile(full):
            sized.append((os.path.getsize(full), rel))
    return sorted(sized, reverse=True)


def run():
    cfg = load_config()

    print("1) index_meta.warn_if_stale() — the guard that is never called:")
    hits = []
    for folder, _, names in os.walk(PROJECT_DIR):
        if "__pycache__" in folder:
            continue
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8", errors="replace") as f:
                if "warn_if_stale" in f.read():
                    hits.append(os.path.relpath(path, PROJECT_DIR))
    print(f"  files mentioning warn_if_stale: {hits}")
    print("  Only its own definition. main.py, app.py and rag_pipeline.py never call it,")
    print("  so a stale index is served without any warning.")

    print("\n2) ...and it would crash if it were called:")
    print("  index_meta.get_current_state() returns {'files': ..., 'settings': ...}")
    print("  index_meta.find_problems() line 54:")
    print('      if saved.get("file") != now["file"]:')
    import src.index_meta as index_meta  # noqa: E402  (PROJECT_DIR is on sys.path)
    try:
        index_meta.find_problems()
        print("  -> ran without error")
    except Exception as error:
        print(f"  -> raises {type(error).__name__}: {error}   ('file' vs 'files')")
    print("  The next line would fail too: os.path.basename(config.SOURCE_FILE) with")
    print(f"  SOURCE_FILE being a list of {len(cfg.SOURCE_FILE)} paths.")

    print("\n3) build_index.py reuses whatever embeddings file it finds:")
    for line in read_source("build_index.py").splitlines():
        if "EMBEDDINGS_FILE" in line and "exists" in line:
            print(f"  {line.strip()}")
    print("  If chunks.json is rebuilt with different settings, the old embeddings.npy")
    print("  is loaded again. FAISS then indexes vectors that do not match the chunk")
    print("  store, and every retrieval returns the wrong chunk for its score.")

    meta_path = cfg.INDEX_META_FILE
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        n_meta = meta.get("n_chunks")
        store = cfg.CHUNK_STORE_FILE
        print(f"\n  index_meta.json records n_chunks = {n_meta}")
        if os.path.exists(store):
            with open(store, encoding="utf-8") as f:
                n_store = len(json.load(f))
            print(f"  chunk_store.json actually holds  = {n_store}")
            print(f"  in sync: {n_meta == n_store}")

    print("\n4) BM25 cache is loaded without validating it against the chunks:")
    for line in read_source("src/hybrid_retriever.py").splitlines():
        if "BM25_INDEX_FILE" in line and "exists" in line:
            print(f"  {line.strip()}")
    print("  load_bm25(chunks) ignores its own argument when the pickle exists.")

    vector_db = os.path.join(PROJECT_DIR, "vector_db")
    if os.path.isdir(vector_db):
        print("\n  Files in vector_db/:")
        for name in sorted(os.listdir(vector_db)):
            size = os.path.getsize(os.path.join(vector_db, name))
            marker = ""
            if name == "bm25.pkl":
                marker = "  <- orphan; config points at bm25_index.pkl"
            print(f"    {size / 1024 / 1024:8.2f} MB  {name}{marker}")

    print("\n5) Repository hygiene — generated artifacts are committed:")
    root = repo_root()
    print(f"  .gitignore present: {os.path.exists(os.path.join(root, '.gitignore'))}")
    sized = tracked_sizes()
    if sized:
        print("  Largest files tracked by git:")
        for size, rel in sized[:6]:
            over = "  <- above GitHub's 100 MB file limit" if size > 100 * 1024 * 1024 else ""
            print(f"    {size / 1024 / 1024:8.2f} MB  {rel}{over}")
        pyc = sum(1 for _, rel in sized if rel.endswith(".pyc"))
        print(f"  Compiled .pyc files tracked: {pyc}")

    print("\nCause: the index is treated as source. Nothing verifies that chunks,")
    print("embeddings and the BM25 pickle were built from the same data, the guard that")
    print("was written for exactly that is dead code with a typo in it, and the outputs")
    print("are versioned instead of regenerated.")
    print("Fix: call warn_if_stale() at startup, compare 'files' not 'file', add a")
    print("--rebuild flag to build_index.py, store the chunk count alongside the BM25")
    print("pickle, and add a .gitignore for outputs/, vector_db/ and __pycache__/.")
