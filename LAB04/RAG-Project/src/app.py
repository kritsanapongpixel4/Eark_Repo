import streamlit as st
from src.rag_pipeline import RAGPipeline
import config

st.set_page_config(page_title="Automotive RAG Chatbot", page_icon="🚗")
st.title("🚗 Automotive Knowledge Assistant (RAG + LM API)")

# 1. โหลด RAGPipeline โมดูลหลัก
@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

# 2. จัดการ Session State สำหรับประวัติแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. รับคำถามจากผู้ใช้
if prompt := st.chat_input("ถามคำถามเกี่ยวกับรถยนต์ สเปก ระบบขับเคลื่อน หรือ EV ได้ที่นี่..."):
    # แสดงคำถามผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. ประมวลผลผ่าน RAG Pipeline (BM25 Hybrid + LM API)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและเรียบเรียงคำตอบ..."):
            result = rag.ask(prompt, top_k=config.TOP_K)
            answer = result.get("answer", "ขออภัย ไม่พบข้อมูลในระบบ")
            st.markdown(answer)

            # แสดงอ้างอิง Chunks ที่ดึงมาได้
            with st.expander("ดูแหล่งอ้างอิงข้อมูล (Retrieved Sources - BM25 & Dense)"):
                for i, chunk in enumerate(result.get("retrieved", []), 1):
                    st.write(f"**[{i}] RRF Score:** {chunk.get('score', 0):.4f} | **BM25:** {chunk.get('bm25_score', 'N/A')} | **Dense:** {chunk.get('dense_score', 'N/A')}")
                    st.write(f"**Q:** {chunk.get('question', '')}")
                    st.caption(chunk.get("answer") or chunk.get("text", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})