```python
import streamlit as st
import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ==============================
# 1. UI CONFIG
# ==============================
st.set_page_config(
    page_title="MFU Research Grant Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 ผู้ช่วยตอบคำถามทุนวิจัย (มฟล.)")
st.markdown("ระบบนี้ใช้ AI ค้นหาคำตอบจากเอกสารประกาศมหาวิทยาลัยแม่ฟ้าหลวง")

# ==============================
# 2. API KEY (SECURE)
# ==============================
try:
    os.environ["AIzaSyC3eJ3T7uoz9NG_FAaM-NYJt--74Q1MhNI"] = st.secrets["AIzaSyC3eJ3T7uoz9NG_FAaM-NYJt--74Q1MhNI"]
except:
    st.warning("⚠️ กรุณาตั้งค่า GOOGLE_API_KEY ใน Streamlit Secrets")
    st.stop()

# ==============================
# 3. TEXT CLEANING FUNCTION
# ==============================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # ลบ space ซ้ำ
    text = text.replace("\n", " ")
    return text.strip()

# ==============================
# 4. RAG SYSTEM
# ==============================
@st.cache_resource(show_spinner="กำลังวิเคราะห์เอกสาร...")
def initialize_rag():

    file_path = "extracted_ocr_data.txt"

    if not os.path.exists(file_path):
        st.error(f"ไม่พบไฟล์ {file_path}")
        st.stop()

    # Load
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Clean OCR text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
        max_output_tokens=512
    )

    # Prompt
    system_prompt = (
        "คุณเป็นเจ้าหน้าที่มหาวิทยาลัยแม่ฟ้าหลวง\n"
        "ตอบโดยใช้เฉพาะข้อมูลใน Context เท่านั้น\n"
        "ห้ามเดาหรือสร้างข้อมูลเอง\n"
        "ถ้าไม่มีข้อมูล ให้ตอบว่า: 'ขออภัยครับ ไม่มีข้อมูลในประกาศ'\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    return rag_chain


# ==============================
# INIT SYSTEM
# ==============================
try:
    rag_chain = initialize_rag()
except Exception as e:
    st.error(f"❌ โหลดระบบไม่สำเร็จ: {e}")
    st.stop()

# ==============================
# 5. CHAT MEMORY
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# 6. USER INPUT
# ==============================
if query := st.chat_input("ถามเกี่ยวกับทุนวิจัย..."):

    # user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # assistant
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาคำตอบ..."):
            try:
                response = rag_chain.invoke({"input": query})

                answer = response.get("answer", "ไม่พบคำตอบ")
                context = response.get("context", [])

                st.markdown(answer)

                # Save answer
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # Show sources
                if context:
                    with st.expander("🔍 แหล่งข้อมูลที่ใช้"):
                        for i, doc in enumerate(context):
                            st.write(f"📄 Chunk {i+1}")
                            st.caption(doc.page_content[:300] + "...")

                else:
                    st.warning("⚠️ ไม่พบข้อมูลที่เกี่ยวข้อง")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

# ==============================
# 7. SIDEBAR
# ==============================
with st.sidebar:
    st.header("📌 ข้อมูลสรุป")

    st.write("- 💰 วงเงิน: ไม่เกิน 50,000 บาท")
    st.write("- ⏱ ระยะเวลา: ไม่เกิน 12 เดือน")
    st.write("- 💵 การจ่าย: 50% / 30% / 20%")

    st.divider()

    if st.button("🗑️ ล้างแชท"):
        st.session_state.messages = []
        st.rerun()
```
