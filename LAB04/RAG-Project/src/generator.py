


# generator.py
# Generate answers with an LLM from retrieved documents.
# Disable USE_LLM to return retrieved text only.

import json
import os
import re
import urllib.request
import urllib.error

from openai import OpenAI

import config
from src.prompt_templates import build_messages


class LLM:
    """เรียก Cloud LLM (รองรับทั้ง Google Gemini API ตรง และ OpenAI / Groq / OpenRouter)"""

    def __init__(self, provider=None):
        self.provider = provider or config.LLM_PROVIDER
        base_url, default_model, key_name = config.LLM_PROVIDERS.get(
            self.provider, config.LLM_PROVIDERS["gemini"]
        )

        self.model = config.LLM_MODEL or default_model
        api_key = os.getenv(key_name) if key_name else ""
        if not api_key and key_name and hasattr(config, key_name):
            api_key = getattr(config, key_name, "")

        self.api_key = api_key.strip()
        self.base_url = base_url

        if self.provider != "gemini":
            self.client = OpenAI(base_url=base_url, api_key=self.api_key or "no-key")
        else:
            self.client = None

    def _call_gemini(self, prompt, timeout=15.0):
        """เรียก Gemini REST API โดยตรง รวดเร็วและแม่นยำ ไม่ต้องพึ่งพา Proxy"""
        models_to_try = [self.model, "gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]
        # ลบโมเดลซ้ำ
        models_to_try = list(dict.fromkeys(models_to_try))

        last_error = None
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": config.LLM_TEMPERATURE,
                    "maxOutputTokens": config.LLM_MAX_TOKENS,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            self.model = m
                            return parts[0].get("text", "").strip()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                last_error = f"HTTP {e.code}: {err_body}"
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(last_error or "Failed to call Gemini API")

    def check_connection(self):
        """ตรวจสอบว่าสามารถเชื่อมต่อและใช้งาน LM API ได้จริงหรือไม่"""
        if not self.api_key:
            return False, "ยังไม่ได้ใส่ API Key ใน config.py"

        try:
            if self.provider == "gemini":
                ans = self._call_gemini("Ping! ตอบสั้นๆว่า OK", timeout=10.0)
                if ans:
                    return True, f"เชื่อมต่อสำเร็จ (Google Gemini - {self.model})"
            else:
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    timeout=10.0,
                )
                return True, f"เชื่อมต่อสำเร็จ ({self.provider} API - {self.model})"
        except Exception as e:
            return False, f"ไม่สามารถเชื่อมต่อ {self.provider} API ({self.model}): {str(e)}"

        return False, f"ไม่สามารถเชื่อมต่อ {self.provider} API"

    def chat(self, messages):
        if self.provider == "gemini":
            # รวม System และ User messages เป็น Prompt สำหรับ Gemini
            full_prompt = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    full_prompt += f"[คำสั่งระบบ]: {content}\n\n"
                elif role == "user":
                    full_prompt += f"{content}\n"
                elif role == "assistant":
                    full_prompt += f"[คำตอบเดิม]: {content}\n"
            return self._call_gemini(full_prompt.strip())

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()


class NoLLM:
    model = "No-LLM Fallback (Extracted Context)"

    def check_connection(self):
        return False, "ยังไม่ได้ใส่ API Key (กรุณาใส่ GEMINI_API_KEY หรือ OPENAI_API_KEY ใน config.py)"

    def chat(self, messages):
        user_message = messages[-1]["content"]

        # สกัดเนื้อหาอ้างอิงออกมาตอบเป็นคำตอบมาตรฐาน
        if "ข้อมูลอ้างอิง:" in user_message:
            parts = user_message.split("ข้อมูลอ้างอิง:")
            context_part = parts[1].split("คำถามของผู้ใช้:")[0].strip()
            if context_part:
                lines = [line.strip() for line in context_part.split("\n\n") if line.strip()]
                if lines:
                    formatted_ans = "จากการสืบค้นฐานข้อมูลยานยนต์ ได้ข้อมูลดังนี้:\n\n"
                    for idx, line in enumerate(lines[:3], start=1):
                        formatted_ans += f"{line}\n"
                    return formatted_ans.strip()

        return config.NO_CONTEXT_MESSAGE


def get_llm():
    if not config.USE_LLM:
        return NoLLM()

    # ลองตามลำดับ: 1. provider ใน config 2. Gemini/OpenAI/Groq ถ้าระบุไว้ 3. Fallback
    providers_to_try = [config.LLM_PROVIDER]
    if (os.getenv("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", "")) and "gemini" not in providers_to_try:
        providers_to_try.insert(0, "gemini")
    if (os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", "")) and "openai" not in providers_to_try:
        providers_to_try.insert(0, "openai")
    if (os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", "")) and "groq" not in providers_to_try:
        providers_to_try.insert(0, "groq")

    for provider in providers_to_try:
        try:
            llm_inst = LLM(provider=provider)
            is_ok, msg = llm_inst.check_connection()
            if is_ok:
                print(f"[LLM Status] {msg}")
                return llm_inst
            else:
                print(f"[LLM Status] {msg}")
        except Exception as error:
            print(f"[LLM Error] {provider}: {error}")

    print("[LLM Status] ⚠️ ไม่สามารถเชื่อมต่อกับบริการ LM ใด ๆ ได้ — ใช้โหมดสกัดคำตอบอ้างอิงชั่วคราว (Fallback Mode)")
    return NoLLM()


class Generator:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def generate(self, question, chunks, history=""):
        # ค้นไม่เจออะไรเลย — ตอบว่าไม่รู้ ดีกว่าให้ LLM เดา
        if not chunks:
            return {
                "answer": config.NO_CONTEXT_MESSAGE,
                "sources": [],
                "no_context": True,
                "llm_connected": False,
            }

        messages = build_messages(question, chunks, history)
        is_connected, status_msg = self.llm.check_connection()

        try:
            if is_connected:
                answer = self.llm.chat(messages)
            else:
                answer = self.llm.chat(messages)
        except Exception as error:
            print(f"[LLM Generate Error] {error}")
            # Fallback format: ใช้เนื้อหาจาก chunk ที่ค้นได้ดีที่สุดพร้อมอ้างอิง [1]
            top_answer = chunks[0].get("answer") or chunks[0].get("text", "")
            answer = f"ข้อมูลที่เกี่ยวข้องจากฐานข้อมูลยานยนต์ [1]:\n{top_answer}"

        if config.DISCLAIMER not in answer:
            answer = f"{answer}\n\n{config.DISCLAIMER}"

        return {
            "answer": answer.strip(),
            "sources": self.build_sources(chunks),
            "no_context": False,
            "llm_connected": is_connected,
            "llm_status": status_msg,
        }

    def build_sources(self, chunks):
        sources = []
        for number, chunk in enumerate(chunks, start=1):
            sources.append({
                "n": number,
                "chunk_id": chunk.get("chunk_id", f"c{number}"),
                "question": chunk.get("question", ""),
                "line_no": chunk.get("line_no", 0),
                "score": round(float(chunk.get("score", 0.0)), 4),
                "bm25_score": round(float(chunk.get("bm25_score", 0.0)), 4) if chunk.get("bm25_score") is not None else None,
                "dense_score": round(float(chunk.get("dense_score", 0.0)), 4) if chunk.get("dense_score") is not None else None,
            })
        return sources
