# -*- coding: utf-8 -*-
# Problem 05: Metadata — every chunk carries metadata fields, but for 99% of the
# knowledge base those fields hold the same placeholder value, so no metadata
# filter can narrow a search.
#
# The project stores category / question / line_no / source per chunk. In DL-05
# the metadata (category, language register) was what made filtering possible.
# Here the fallback loader writes one constant category for 60k chunks.
import collections

from data_loader import generic_chunks, load_chunks, oneline, parsed_chunks


def filter_by_category(chunks, keyword):
    return [c for c in chunks if keyword.lower() in c["category"].lower()]


def run():
    chunks = load_chunks()
    generic = generic_chunks(chunks)
    parsed = parsed_chunks(chunks)

    categories = collections.Counter(c["category"] for c in chunks)
    print(f"Distinct categories: {len(categories)} across {len(chunks)} chunks")
    print("Largest categories:")
    for name, count in categories.most_common(5):
        share = count / len(chunks) * 100
        print(f"  {count:>6}  ({share:5.1f}%)  {name}")

    print("\nThe metadata stored on a chunk from the generic loader:")
    sample = generic[500]
    for key in ("chunk_id", "qa_id", "category", "question", "part_idx", "line_no"):
        print(f"  {key:<10} = {oneline(sample.get(key), 60)!r}")
    print("  -> only line_no distinguishes it from the other 60,010 chunks")

    print("\nMetadata filtering that DOES work (parsed part of the KB):")
    for brand in ("Toyota", "BMW", "Lexus"):
        hits = filter_by_category(parsed, brand)
        print(f"  category contains '{brand}': {len(hits)} chunks")
        if hits:
            print(f"      e.g. {oneline(hits[0]['question'], 70)}")

    print("\nThe same filter on the 6000-vehicle dataset, which is where the specs live:")
    for brand in ("Toyota", "BMW", "Kia"):
        hits = filter_by_category(generic, brand)
        print(f"  category contains '{brand}': {len(hits)} chunks")
    print("  -> 0 every time; the brand exists in the text but never in the metadata")

    print("\nQuery: 'Show me only Kia models under EUR 60,000'")
    print("  Needs a filter on brand + price. Available metadata: category='General Knowledge',")
    print("  question='Automotive information from CARS_DATASET_RAG.txt', line_no=<int>.")
    print("  Neither brand nor price can be filtered, so the query degrades to pure")
    print("  keyword/vector search over 60k one-line fragments.")

    print("\nNote: document_loader.load_xlsx() already extracts Brand, Segment, Price,")
    print("Horsepower and more into structured fields — but config.SOURCE_FILE lists only")
    print("the three .txt files, so CARS_DATASET.xlsx and that parser are never used.")

    print("\nCause: metadata is written by whichever loader branch runs, and the branch")
    print("that handles 99% of the data writes constants.")
    print("Fix: parse brand / segment / powertrain / price into chunk metadata (the xlsx")
    print("loader already shows how), then filter on those fields before scoring.")
