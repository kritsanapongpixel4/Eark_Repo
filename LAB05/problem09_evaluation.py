# -*- coding: utf-8 -*-
# Problem 09: Evaluation — the evaluation suite cannot run at all, and even if
# it could, the golden set it builds would not measure this knowledge base.
#
#     python -m evaluation.build_golden_set   -> AttributeError: GOLDEN_SET_SIZE
#     python -m evaluation.eval_retrieval     -> golden_set.json missing, exit 1
import collections
import os
import re

from data_loader import generic_chunks, load_chunks, load_config, oneline, read_source


def run():
    cfg = load_config()
    chunks = load_chunks()

    print("Evaluation settings that DO exist in config.py:")
    for name in ("GOLDEN_SET_FILE", "EVAL_RETRIEVAL_FILE", "EVAL_GENERATION_FILE", "EVAL_K_VALUES"):
        value = getattr(cfg, name)
        print(f"  {name:<22} = {os.path.basename(str(value)) if 'FILE' in name else value}")

    print("\nStep 1 — build the golden set:")
    print("  evaluation/build_golden_set.py line 97:")
    print("      per_category = max(1, config.GOLDEN_SET_SIZE // len(by_category))")
    print(f"  config.GOLDEN_SET_SIZE exists: {hasattr(cfg, 'GOLDEN_SET_SIZE')}")
    print("  -> AttributeError before a single test item is written.")

    print("\nStep 2 — run the retrieval evaluation:")
    exists = os.path.exists(cfg.GOLDEN_SET_FILE)
    print(f"  outputs/golden_set.json exists: {exists}")
    print("  eval_retrieval.load_golden_set() raises SystemExit(1) when it is missing,")
    print("  so no retrieval metric has ever been produced for this project.")

    print("\nStep 3 — what the golden set would contain if it did run:")
    source = read_source("evaluation/build_golden_set.py")
    block = re.search(r"TO_SLANG\s*=\s*\{(.*?)\}", source, re.S)
    if block:
        pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block.group(1))
        print(f"  TO_SLANG rewrites {len(pairs)} medical terms into Thai everyday words:")
        for formal, slang in pairs[:4]:
            print(f"    {formal} -> {slang}")
    print("  The 'slang' query variant is built from those rules. On an English")
    print("  automotive KB none of them fire, so the slang variant equals the verbatim")
    print("  variant and the hardest test case measures nothing.")

    print("\nStep 4 — the sampling problem, measured on the real chunk store:")
    generic = generic_chunks(chunks)
    by_category = collections.Counter(c["category"] for c in chunks)
    print(f"  build_golden_set() samples per category. Categories: {len(by_category)}")
    print(f"  One category ('General Knowledge') holds {len(generic)} chunks ({len(generic) / len(chunks) * 100:.1f}%)")
    print("  and every one of them carries the same question text:")
    print(f"    {oneline(generic[0]['question'], 70)!r}")
    print("  Ground truth is 'the query must retrieve chunk_id X'. With 60,011 chunks")
    print("  sharing one question, any of them is an equally good match, so Hit@k and")
    print("  MRR on that category score near zero no matter how good the retriever is.")

    print("\n  Illustration — Hit@k for a query whose gold chunk is one of N identical ones:")
    for k in cfg.EVAL_K_VALUES:
        chance = min(1.0, k / len(generic))
        print(f"    Top-{k}: probability of hitting the exact gold chunk_id = {chance:.6f}")

    print("\nCause: evaluation was copied from the previous project and never adapted.")
    print("Two config keys are missing, the query-variant generator targets the wrong")
    print("language, and the ground truth is unidentifiable because of problem 03.")
    print("Fix: add GOLDEN_SET_SIZE, rebuild TO_SLANG from automotive vocabulary, and")
    print("score a retrieval as correct when it returns the right VEHICLE RECORD rather")
    print("than one exact chunk_id.")
