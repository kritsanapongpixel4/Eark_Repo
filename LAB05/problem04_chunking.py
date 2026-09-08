# -*- coding: utf-8 -*-
# Problem 04: Chunking — CHUNK_SIZE / CHUNK_OVERLAP are configured but almost
# never used, because the loader already cut the data far too small.
#
# text_splitter.split_text() only splits when a text is longer than CHUNK_SIZE.
# The generic loader emits one line per chunk, so nearly every chunk is already
# well under 600 characters and the overlap setting has nothing to protect.
import statistics

from data_loader import generic_chunks, load_chunks, load_config, oneline


def run():
    cfg = load_config()
    chunks = load_chunks()
    generic = generic_chunks(chunks)

    lengths = [len(c["text"]) for c in chunks]
    split = [c for c in chunks if c["part_idx"] > 0]

    print("Chunking settings in config.py:")
    print(f"  CHUNK_SIZE    = {cfg.CHUNK_SIZE} characters")
    print(f"  CHUNK_OVERLAP = {cfg.CHUNK_OVERLAP} characters")

    print("\nActual chunk sizes produced:")
    print(f"  count   = {len(lengths)}")
    print(f"  min     = {min(lengths)}")
    print(f"  median  = {statistics.median(lengths):.0f}")
    print(f"  mean    = {statistics.mean(lengths):.0f}")
    print(f"  max     = {max(lengths)}")
    print(f"  shorter than 150 chars: {sum(1 for n in lengths if n < 150)} ({sum(1 for n in lengths if n < 150) / len(lengths) * 100:.1f}%)")

    print(f"\nChunks that were actually long enough to be split: {len(split)} out of {len(chunks)}")
    print(f"  -> CHUNK_OVERLAP={cfg.CHUNK_OVERLAP} is applied to {len(split) / len(chunks) * 100:.3f}% of the data")

    print("\nWhy that matters — one vehicle record spread across separate chunks:")
    record = None
    for i, c in enumerate(generic):
        if c["answer"].startswith("Record 1:"):
            record = i
            break
    if record is not None:
        for c in generic[record:record + 7]:
            print(f"  chunk_id={c['chunk_id']:<6} {oneline(c['answer'], 72)}")

        print("\n  Query: 'What is the price of the Kia R7 Pro?'")
        print("  The model name lives in one chunk and the price in another. Retrieval can")
        print("  return either one, but no single chunk answers the question, and nothing")
        print("  in the price chunk says which car it belongs to:")
        for c in generic[record:record + 4]:
            if c["answer"].startswith("Price:"):
                print(f"      {c['answer']!r}  <- no brand, no model, unusable as evidence")

    print("\nWhat a correctly sized chunk looks like (from the glossary parser):")
    for c in chunks[:2]:
        print(f"  [{len(c['text'])} chars] {oneline(c['text'], 100)}")

    print("\nCause: chunk size is decided by the loader, not by text_splitter.py.")
    print("Once records are shattered per line, no CHUNK_SIZE setting can put them")
    print("back together, and each chunk is too small to be self-contained.")
    print("Fix: make the loader emit one chunk per vehicle record (roughly 400-600")
    print("chars), which is exactly the size CHUNK_SIZE was tuned for.")
