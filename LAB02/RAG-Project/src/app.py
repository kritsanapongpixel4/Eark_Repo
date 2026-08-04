import streamlit as st
import requests  # สำหรับต่อ API ของ Ollama (Local LLM)
from src.retriever import Retriever  # ดึง Retriever จากโปรเจกต์เดิมมาใช้
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME

st.set_page_config(page_title="Pad Krapao Local Chatbot", page_icon="🍳")
st.title("🍳 ผัดกะเพรา Q&A Assistant (Local RAG)")

# 1. โหลด Retriever โมดูลเดิมขึ้นมารอบแรก
@st.cache_resource
def load_retriever():
    return Retriever(vector_db_dir=VECTOR_DB_DIR, model_name=EMBEDDING_MODEL_NAME)

retriever = load_retriever()

# 2. จัดการ Session State สำหรับประวัติแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. รับคำถามจากผู้ใช้
if prompt := st.chat_input("ถามคำถามเกี่ยวกับผัดกะเพราได้ที่นี่..."):
    # แสดงคำถามผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. ดึงข้อมูลบริบทจาก RAG (Retrieval)
    top_chunks = retriever.retrieve(prompt, top_k=3)
    context_text = "\n\n".join([chunk["text"] for chunk in top_chunks])

    # 5. สร้าง Prompt รวมสำหรับส่งให้ Local LLM
    full_prompt = f"""คุณคือผู้เชี่ยวชาญด้านอาหาร จงตอบคำถามต่อไปนี้โดยใช้ข้อมูลบริบทที่กำหนดให้เท่านั้น

ข้อมูลบริบท (Context):
{context_text}

คำถาม: {prompt}
คำตอบ:"""

    # 6. ส่งไปประมวลผลที่ Local LLM (ตัวอย่าง Ollama)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและเรียบเรียงคำตอบ..."):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5",  # หรือชื่อโมเดลที่คุณโหลดไว้ใน Ollama
                        "prompt": full_prompt,
                        "stream": False
                    }
                ).json()
                answer = response.get("response", "ไม่สามารถสร้างคำตอบได้")
            except Exception as e:
                answer = f"⚠️ ไม่สามารถเชื่อมต่อกับ Local LLM ได้: {e}"

            st.markdown(answer)
            
            # (Optional) แสดงอ้างอิง Chunks ที่ดึงมาได้
            with st.expander("ดูแหล่งอ้างอิงข้อมูล (Retrieved Chunks)"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.write(f"**[{i}] Score:** {chunk.get('score', 0):.4f}")
                    st.caption(chunk["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})