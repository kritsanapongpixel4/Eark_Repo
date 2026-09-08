# -*- coding: utf-8 -*-
# Problem 07: Generation — retrieval can be correct, but the generation stage
# has a dead branch, an extra API round-trip per question, and it answers an
# English knowledge base in Thai.
import re

from data_loader import load_chunks, load_config, oneline, read_source


def show_dead_branch():
    source = read_source("src/generator.py")
    block = re.search(r"( *)is_connected, status_msg = self\.llm\.check_connection\(\).*?answer = f\"", source, re.S)
    if not block:
        return
    for line in block.group(0).splitlines()[:9]:
        print(f"  {line}")


def run():
    cfg = load_config()
    chunks = load_chunks()

    print("1) Dead branch in Generator.generate() — both sides do the same thing:")
    show_dead_branch()
    print("  The if/else was meant to switch to an offline fallback when the LLM is")
    print("  unreachable. Both branches call self.llm.chat(messages), so the check")
    print("  only decides what llm_connected reports to the UI.")

    print("\n2) check_connection() is called once per question, before every answer:")
    print("  generator.generate()  -> self.llm.check_connection()")
    print("  For the gemini provider that sends a real 'Ping!' request to the API.")
    print("  app.py /api/status calls it again on every status poll.")
    print(f"  Cost: 2 API calls and up to {10.0:.0f}s extra latency per user question,")
    print("  purely to fill in a status string.")

    print("\n3) Language mismatch between the context and the instructions:")
    source = read_source("src/prompt_templates.py")
    system = re.search(r'SYSTEM_PROMPT = """(.*?)"""', source, re.S)
    if system:
        print(f"  SYSTEM_PROMPT (Thai): {system.group(1).strip().splitlines()[0]}")
    if chunks:
        print(f"  Retrieved context (English): {oneline(chunks[0]['answer'], 80)}")
    print(f"  NO_CONTEXT_MESSAGE: {cfg.NO_CONTEXT_MESSAGE}")
    print(f"  DISCLAIMER: {cfg.DISCLAIMER}")
    print("  An English question about an English KB is answered in Thai, and rule 6")
    print("  ('answer in at most 5-6 sentences') truncates multi-spec comparisons.")

    print("\n4) Faithfulness risk from problem 04 — the context sent to the model:")
    generic = [c for c in chunks if c["category"] == "General Knowledge"]
    price_chunk = next((c for c in generic if c["answer"].startswith("Price:")), None)
    perf_chunk = next((c for c in generic if c["answer"].startswith("Performance:")), None)
    if price_chunk and perf_chunk:
        print("  Question: 'How much does the Kia R7 Pro cost and how fast is it?'")
        print("  format_context() would build a prompt like:")
        print(f"      [1] {oneline(price_chunk['answer'], 70)}")
        print(f"      [2] {oneline(perf_chunk['answer'], 70)}")
        print("  Neither block names a car. The model must either refuse, or attach the")
        print("  numbers to whichever model the question mentioned — a grounded-looking")
        print("  answer with no evidence that these figures belong to that vehicle.")

    print("\n5) The offline fallback silently drops the citation numbering:")
    nollm = re.search(r"formatted_ans = \"(.*?)\"", read_source("src/generator.py"))
    if nollm:
        print(f"  NoLLM.chat() returns: {nollm.group(1)}")
    print("  followed by raw chunk text with no [1] [2] markers, even though the system")
    print("  prompt promises inline citations. build_sources() still reports sources,")
    print("  so the UI shows citations that the answer text never refers to.")

    print("\nCause: the generation stage was carried over from the previous project and")
    print("its error handling degrades silently instead of failing loudly.")
    print("Fix: delete the dead branch, cache check_connection() at startup instead of")
    print("per request, translate the prompt templates to the KB language, and make the")
    print("fallback emit the same [n] citations as the LLM path.")
