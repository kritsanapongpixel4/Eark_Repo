# -*- coding: utf-8 -*-
# Problem 01: Hallucination — the "no relevant data" guard can never fire
#
# rag_pipeline.py clears the retrieved chunks when the best RRF score is below
# config.RELEVANCE_SCORE_THRESHOLD, so the generator answers "not found" instead
# of guessing. In this project that guard is dead code: the threshold is set
# below the smallest score RRF can ever produce.
from data_loader import load_chunks, load_config, oneline


def rrf_score(rank, n_lists=1, rrf_k=60):
    # The exact formula used in hybrid_retriever.reciprocal_rank_fusion()
    return sum(1.0 / (rrf_k + rank) for _ in range(n_lists))


def run():
    cfg = load_config()
    chunks = load_chunks()

    rrf_k = cfg.RRF_K
    candidate_k = cfg.CANDIDATE_K
    threshold = cfg.RELEVANCE_SCORE_THRESHOLD

    worst = rrf_score(candidate_k, 1, rrf_k)          # last candidate, found by one retriever only
    best = rrf_score(1, 2, rrf_k)                     # rank 1 in both BM25 and dense

    print("Settings read from LAB04/RAG-Project/config.py:")
    print(f"  RRF_K                       = {rrf_k}")
    print(f"  CANDIDATE_K                 = {candidate_k}")
    print(f"  RELEVANCE_SCORE_THRESHOLD   = {threshold}")

    print("\nRange of RRF scores the retriever can actually return:")
    print(f"  highest possible = 2 x 1/({rrf_k}+1)  = {best:.5f}")
    print(f"  lowest  possible = 1/({rrf_k}+{candidate_k}) = {worst:.5f}")

    print(f"\nIs any score below the threshold? {worst < threshold}")
    print(f"  lowest score {worst:.5f} is {worst / threshold:.1f}x LARGER than the threshold {threshold}")

    print("\nWhat that means in practice:")
    print("  Every query — even one with nothing to do with cars — still returns")
    print(f"  TOP_K = {cfg.TOP_K} chunks with a score above the threshold, so this branch")
    print("  in rag_pipeline.ask() never runs:")
    print("      if best_score < config.RELEVANCE_SCORE_THRESHOLD:")
    print("          chunks = []   # <- unreachable")

    print("\nExample of an out-of-scope query:")
    query = "How much is a flight ticket to Chiang Mai?"
    print(f"  Query: {query}")
    print(f"  Guard should trigger -> answer '{cfg.NO_CONTEXT_MESSAGE}'")
    print(f"  Guard actually triggers: NO. {cfg.TOP_K} unrelated car chunks are sent to the LLM,")
    print("  and the prompt then asks the model to answer from that context.")

    if chunks:
        print("\nThe kind of chunk that gets sent as 'evidence':")
        for c in chunks[4:7]:
            print(f"  [{c['category']}] {oneline(c['answer'], 70)}")

    print("\nCause: RRF scores are bounded below by 1/(RRF_K + CANDIDATE_K), but the")
    print("threshold was tuned as if the scores were cosine similarities (0.0-1.0).")
    print("Fix: set the threshold relative to the RRF range (e.g. 0.02), or filter on")
    print("dense_score instead, which really is a cosine similarity.")
