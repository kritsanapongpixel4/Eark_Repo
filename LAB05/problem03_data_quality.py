# -*- coding: utf-8 -*-
# Problem 03: Data Quality — the biggest data file falls through to the generic
# fallback loader, so 99% of the knowledge base is unparsed noise.
#
# document_loader.load_qa_file() dispatches on the file name. There is a parser
# for eu_and_jp_cars_dataset.txt and one for car_fundamentals_glossary_kb.txt,
# but CARS_DATASET_RAG.txt (4 MB, 6000 vehicle records) matches neither, so it
# hits the "Generic text loader" branch: one record per LINE of the file.
import collections
import re

from data_loader import generic_chunks, load_chunks, oneline, parsed_chunks, read_data_lines


def run():
    chunks = load_chunks()
    generic = generic_chunks(chunks)
    parsed = parsed_chunks(chunks)

    print(f"Total chunks in the knowledge base: {len(chunks)}")
    print(f"  parsed by a real format-aware loader : {len(parsed):>6}  ({len(parsed) / len(chunks) * 100:.1f}%)")
    print(f"  produced by the generic line loader  : {len(generic):>6}  ({len(generic) / len(chunks) * 100:.1f}%)")

    questions = collections.Counter(c["question"] for c in chunks)
    print(f"\nDistinct questions across {len(chunks)} chunks: {len(questions)}")
    print("Most repeated question:")
    top_question, top_count = questions.most_common(1)[0]
    print(f"  {top_count} chunks all share the question {top_question!r}")

    print("\nWhat one vehicle record looks like in the source file:")
    for line in read_data_lines("CARS_DATASET_RAG.txt")[16:24]:
        print(f"  | {line}")

    print("\nHow the generic loader stored it — one independent chunk per line:")
    for c in generic[8:14]:
        print(f"  chunk_id={c['chunk_id']:<6} answer={oneline(c['answer'], 60)!r}")

    print("\nFile-header lines were ingested as knowledge too:")
    for c in generic[:5]:
        print(f"  {oneline(c['answer'], 80)!r}")

    duplicates = collections.Counter(c["question"] for c in parsed)
    repeated = [(q, n) for q, n in duplicates.most_common() if n > 1]
    print(f"\nDuplicates inside the properly parsed part ({len(parsed)} chunks): {len(repeated)} repeated questions")
    for q, n in repeated[:5]:
        print(f"  x{n}  {oneline(q, 70)}")

    numbered = [c for c in parsed if c["question"].startswith("What is ") and c["question"][8].isdigit()]
    print(f"\nGlossary questions that leaked their section number into the text: {len(numbered)}")
    for c in numbered[:3]:
        clean = re.sub(r"^What is [\d.]+\s*", "What is ", c["question"])
        print(f"  {c['question']!r}  ->  should be {clean!r}")

    print("\nCause: load_qa_file() dispatches on file name, and the largest data file")
    print("matches no parser. The silent fallback splits it per line, which destroys")
    print("record boundaries, repeats one meaningless question 60k times, and stores")
    print("the file header as if it were automotive knowledge.")
    print("Fix: write a 'Record N:' parser for CARS_DATASET_RAG.txt, skip header")
    print("blocks, strip the numbering from glossary headings, and make the generic")
    print("fallback raise instead of silently accepting an unknown format.")
