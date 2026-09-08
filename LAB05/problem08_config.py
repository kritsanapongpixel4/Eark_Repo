# -*- coding: utf-8 -*-
# Problem 08: RAG Configuration — the API key is read from an environment
# variable that does not exist, which crashes LLM.__init__ and silently drops
# the whole system into the offline fallback.
#
#     config.py line 71:
#         GEMINI_API_KEY = os.getenv("[ENCRYPTION_KEY]")
#
# There is no environment variable literally named "[ENCRYPTION_KEY]", so the
# value is None rather than "".
import os

from data_loader import load_config


def trace_llm_init(cfg):
    # Replay generator.LLM.__init__ step by step, without importing openai
    base_url, default_model, key_name = cfg.LLM_PROVIDERS.get(
        cfg.LLM_PROVIDER, cfg.LLM_PROVIDERS["gemini"]
    )
    print(f"  provider  = {cfg.LLM_PROVIDER!r}")
    print(f"  key_name  = {key_name!r}")

    api_key = os.getenv(key_name) if key_name else ""
    print(f"  os.getenv({key_name!r})            -> {api_key!r}")

    if not api_key and key_name and hasattr(cfg, key_name):
        api_key = getattr(cfg, key_name, "")
        print(f"  getattr(config, {key_name!r})     -> {api_key!r}")

    print(f"  api_key.strip()                      -> ", end="")
    try:
        stripped = api_key.strip()
        print(repr(stripped))
        return True
    except AttributeError as error:
        print(f"{type(error).__name__}: {error}")
        return False


def run():
    cfg = load_config()

    print("Feature switches in config.py:")
    for name in ("USE_HYBRID", "USE_RERANK", "USE_QUERY_TRANSFORM", "USE_MEMORY", "USE_LLM"):
        print(f"  {name:<22} = {getattr(cfg, name)}")

    print("\nAPI key wiring:")
    print(f'  GEMINI_API_KEY = os.getenv("[ENCRYPTION_KEY]")  -> {cfg.GEMINI_API_KEY!r}')
    print(f"  OPENAI_API_KEY     -> {cfg.OPENAI_API_KEY!r}")
    print(f"  GROQ_API_KEY       -> {cfg.GROQ_API_KEY!r}")
    print(f"  OPENROUTER_API_KEY -> {cfg.OPENROUTER_API_KEY!r}")

    print("\nReplaying generator.LLM.__init__() with the current config:")
    ok = trace_llm_init(cfg)

    if not ok:
        print("\n  get_llm() catches that AttributeError and moves on:")
        print("      except Exception as error:")
        print('          print(f"[LLM Error] {provider}: {error}")')
        print("  With no provider left to try it returns NoLLM(), so USE_LLM = True")
        print("  but no LLM ever runs. Answers come from NoLLM.chat(), which just")
        print("  reprints the retrieved chunks.")

    print("\nConfig values that no code path can honour:")
    print(f"  LLM_TEMPERATURE = {cfg.LLM_TEMPERATURE} and LLM_MAX_TOKENS = {cfg.LLM_MAX_TOKENS}")
    print("      -> ignored by NoLLM, so the fallback answer length is uncontrolled")
    default_model = cfg.LLM_PROVIDERS["gemini"][1]
    print(f"  default gemini model = {default_model!r}")
    print("      -> generator._call_gemini() also hardcodes a second list of models to try;")
    print("         the config default and the hardcoded list can drift apart")

    print("\nSettings referenced in code but missing from config.py:")
    for name in ("RERANK_MODEL_NAME", "GOLDEN_SET_SIZE"):
        print(f"  config.{name:<18} exists: {hasattr(cfg, name)}")

    print("\nCause: a placeholder was left in the key lookup, and every consumer of the")
    print("config treats a missing value as 'feature disabled' instead of 'misconfigured'.")
    print("The switches printed by show_settings() therefore describe the intended")
    print("pipeline, not the running one.")
    print('Fix: os.getenv("GEMINI_API_KEY", "") so the value is always a string, validate')
    print("required settings at startup, and fail loudly when USE_LLM is True but no key")
    print("is usable.")
