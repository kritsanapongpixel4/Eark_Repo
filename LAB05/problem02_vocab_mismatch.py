# -*- coding: utf-8 -*-
# Problem 02: Vocabulary Mismatch — the query-transform layer belongs to a
# different project and a different language than the knowledge base.
#
# LAB04 was forked from an earlier Thai sexual-health RAG project. The knowledge
# base was replaced with English automotive data, but query_transform.py,
# prompt_templates.py and memory.py still carry the old Thai vocabulary.
import re

from data_loader import load_chunks, oneline, read_source

THAI_PATTERN = re.compile(r"[\u0e00-\u0e7f]")

# Vocabulary a real user of a car assistant would type, and the KB term it means
CAR_SLANG = {
    "petrol head": "internal combustion engine enthusiast",
    "pickup": "Pickup Truck",
    "boot space": "Boot Capacity",
    "0-60": "0-100 km/h acceleration",
    "self driving": "ADAS Level",
    "gas mileage": "Efficiency (Wh/km)",
}


def slang_map_from_source():
    # Read the real SLANG_MAP out of src/query_transform.py
    source = read_source("src/query_transform.py")
    block = re.search(r"SLANG_MAP\s*=\s*\{(.*?)\}", source, re.S)
    if not block:
        return {}
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block.group(1))
    return dict(pairs)


def thai_ratio(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if THAI_PATTERN.match(ch)) / len(letters)


def run():
    slang_map = slang_map_from_source()
    chunks = load_chunks()

    print("SLANG_MAP that query_transform.normalize_query() applies to every query:")
    for slang, formal in slang_map.items():
        print(f"  {slang}  ->  {formal}")

    kb_text = " ".join(c["text"] for c in chunks[:2000])
    hits = {s: kb_text.count(s) for s in slang_map}
    print(f"\nOccurrences of those terms in the automotive knowledge base: {sum(hits.values())}")
    print("  -> every rewrite rule is a no-op on this dataset")

    print("\nVocabulary a car user actually types, and what the KB calls it:")
    for user_word, kb_word in CAR_SLANG.items():
        covered = user_word in slang_map
        print(f"  '{user_word}' -> '{kb_word}'  | handled by SLANG_MAP: {covered}")

    print("\nLanguage of the knowledge base vs the prompts:")
    print(f"  chunk text is Thai by  {thai_ratio(kb_text) * 100:.1f}% of its letters")
    system_prompt = read_source("src/prompt_templates.py")
    rewrite = re.search(r"REWRITE_PROMPT = \"\"\"(.*?)\"\"\"", system_prompt, re.S)
    if rewrite:
        first_line = rewrite.group(1).strip().splitlines()[0]
        print(f"  REWRITE_PROMPT is Thai by {thai_ratio(first_line) * 100:.1f}% of its letters")
        print(f"  and it says: {first_line}")
        print("  (translation: 'rewrite the question for searching a SEXUAL HEALTH database')")

    print("\nmemory.is_followup() decides whether to rewrite using Thai particles only:")
    markers = re.search(r"markers = \((.*?)\)", read_source("src/memory.py"), re.S)
    if markers:
        print(f"  markers = ({oneline(markers.group(1), 70)})")
    print("  An English follow-up like 'and what about its range?' matches none of them,")
    print("  so history is never injected and the follow-up is searched without context.")

    print("\nCause: the retrieval-side vocabulary layer was never migrated when the")
    print("knowledge base changed domain and language. Query transformation silently")
    print("does nothing useful, while the prompts push the model toward the wrong domain.")
    print("Fix: rebuild SLANG_MAP from automotive terms, translate the prompt templates")
    print("to the KB language, and detect follow-ups with a language-neutral rule.")
