# -*- coding: utf-8 -*-
# Problem 06: Re-ranking — the reranker can never load, and the failure is
# swallowed, so the pipeline reports "rerank enabled" while doing nothing.
#
# rerankers.Reranker.__init__ reads config.RERANK_MODEL_NAME. That setting does
# not exist in config.py, so the constructor raises AttributeError. get_reranker()
# catches every exception and returns None, which the pipeline treats as
# "reranking is off".
from data_loader import load_chunks, load_config, oneline, read_source

# Terms a first-stage retriever weighs equally, vs terms that really answer the query
GENERIC_TERMS = ["battery", "range", "electric"]
SPECIFIC_TERMS = ["thermal management", "heat pump", "battery health"]


def first_stage(chunk):
    text = (chunk["question"] + " " + chunk["answer"]).lower()
    return sum(text.count(term) for term in GENERIC_TERMS)


def rerank(chunk):
    text = (chunk["question"] + " " + chunk["answer"]).lower()
    return first_stage(chunk) + sum(5 for term in SPECIFIC_TERMS if term in text)


def run():
    cfg = load_config()

    print("Settings that claim reranking is active:")
    print(f"  config.USE_RERANK        = {cfg.USE_RERANK}")
    print(f"  config.CANDIDATE_K       = {cfg.CANDIDATE_K}  (candidates handed to the reranker)")
    print(f"  config.RERANK_MODEL_NAME exists: {hasattr(cfg, 'RERANK_MODEL_NAME')}")

    print("\nWhat happens when get_reranker() runs with USE_RERANK = True:")
    print("  rerankers.py line 24:  config.RERANK_MODEL_NAME")
    try:
        cfg.RERANK_MODEL_NAME
    except AttributeError as error:
        print(f"  -> raises {type(error).__name__}: {error}")
    print("  get_reranker() catches it and prints a warning, then returns None.")
    print("  hybrid_retriever then takes the no-reranker path:")
    print("      keep = config.CANDIDATE_K if self.reranker else top_k")
    print(f"      -> keep = {cfg.TOP_K} instead of {cfg.CANDIDATE_K}: the candidate pool is discarded")
    print("  rag_pipeline.show_settings() still prints rerank as enabled.")

    print("\nCost of not reranking, measured on the real knowledge base:")
    chunks = [c for c in load_chunks() if c["category"] != "General Knowledge"]
    first = sorted(chunks, key=first_stage, reverse=True)[:8]
    second = sorted(first, key=rerank, reverse=True)

    print(f"  Query: 'how do I keep my EV battery healthy in hot weather?'")
    print("\n  First-stage ranking (generic terms weighed equally):")
    for i, c in enumerate(first, start=1):
        print(f"    {i}. score={first_stage(c):>2}  {oneline(c['question'], 62)}")

    print("\n  After reranking (specific terms weighted):")
    for i, c in enumerate(second, start=1):
        print(f"    {i}. score={rerank(c):>2}  {oneline(c['question'], 62)}")

    moved = second[0] != first[0]
    print(f"\n  Did the best document move to rank 1? {moved}")
    if moved:
        old_rank = first.index(second[0]) + 1
        print(f"  It was at rank {old_rank}, outside TOP_K = {cfg.TOP_K}, so the answer never saw it.")

    print("\nA second reason the first stage is noisy — BM25 corpus construction:")
    source = read_source("src/hybrid_retriever.py")
    for line in source.splitlines():
        if "chunk['question']" in line and "chunk['text']" in line:
            print(f"  {line.strip()}")
    print("  chunk['text'] already begins with 'Question: <question> Answer: ...',")
    print("  so the question is indexed three times per chunk. For the 60k chunks that")
    print("  all share one question, that inflates the same tokens across the corpus.")

    print("\nCause: a missing config key turns a feature into a silent no-op, and the")
    print("failure path is indistinguishable from 'the feature is switched off'.")
    print("Fix: add RERANK_MODEL_NAME (e.g. 'BAAI/bge-reranker-base') to config.py, let")
    print("get_reranker() re-raise when USE_RERANK is explicitly True, and build the BM25")
    print("corpus from chunk['text'] alone.")
