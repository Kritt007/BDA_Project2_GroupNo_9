import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. ตั้งค่าหน้าเว็บ (UI Configuration)
st.set_page_config(
    page_title="MFU Research Grant Assistant", 
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 ผู้ช่วยตอบคำถามทุนวิจัยพัฒนาการเรียนรู้ (มฟล.)")
st.markdown("ระบบนี้ใช้ AI ช่วยค้นหาคำตอบจากประกาศมหาวิทยาลัยแม่ฟ้าหลวง เรื่อง หลักเกณฑ์การขอรับสนับสนุนทุนวิจัยฯ")

# 2. การจัดการ API Key
# แนะนำให้ตั้งค่าใน Streamlit Secrets หรือ Environment Variable เพื่อความปลอดภัย
if "GOOGLE_API_KEY" not in st.session_state:
    # คุณสามารถใส่ Key ตรงๆ ได้ที่นี่เพื่อทดสอบ (แต่ไม่แนะนำหากจะนำขึ้น GitHub)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyC3eJ3T7uoz9NG_FAaM-NYJt--74Q1MhNI"

# 3. ฟังก์ชันเตรียมระบบฐานข้อมูล (RAG Initialization)
@st.cache_resource(show_spinner="กำลังวิเคราะห์เอกสารประกาศ...")
def initialize_rag_system():
    file_path = "extracted_ocr_data.txt"
    
    if not os.path.exists(file_path):
        st.error(f"ไม่พบไฟล์ '{file_path}' กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับโค้ด")
        st.stop()

    # โหลดเอกสาร [cite: 1]
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    
    # หั่นข้อความให้เล็กลงเพื่อให้ AI ค้นหาได้แม่นยำขึ้น
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # สร้าง Vector Store สำหรับการค้นหา (Embedding ภาษาไทย-อังกฤษ)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ตั้งค่าตัวโมเดล AI (Gemini 1.5 Flash - เร็วและแม่นยำ)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.2,
    max_retries=2
)   
    # กำหนดคำสั่ง (Prompt) ให้ AI สวมบทบาทเป็นเจ้าหน้าที่กองบริหารงานวิจัย
    system_prompt = (
        "คุณคือเจ้าหน้าที่ผู้เชี่ยวชาญด้านกฎระเบียบทุนวิจัยเพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง "
        "จงตอบคำถามโดยใช้ข้อมูลจากเนื้อหา (Context) ที่ให้มาเท่านั้น "
        "หากในเนื้อหาไม่มีคำตอบ ให้บอกว่า 'ขออภัยครับ ข้อมูลส่วนนี้ไม่มีระบุในประกาศ' "
        "ตอบคำถามด้วยภาษาไทยที่สุภาพ เป็นกันเองแต่เป็นทางการ\n\n"
        "เนื้อหาประกอบการพิจารณา: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # สร้าง Chain สำหรับการตอบคำถาม
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

# เรียกใช้ระบบ
try:
    rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {e}")
    st.stop()

# 4. ส่วนแสดงผลการแชท (Chat UI)
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ช่องรับคำถามจากผู้ใช้
if user_query := st.chat_input("พิมพ์คำถามของคุณที่นี่ เช่น 'ทุนนี้ให้งบเท่าไหร่?' หรือ 'ใครขอทุนได้บ้าง?'"):
    # บันทึกและแสดงคำถามของผู้ใช้
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ให้ AI ประมวลผลคำตอบ
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลจากประกาศ..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                full_response = response["answer"]
                st.markdown(full_response)
                # บันทึกคำตอบของ AI
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"ไม่สามารถตอบคำถามได้ในขณะนี้: {e}")

# 5. แถบด้านข้างสำหรับข้อมูลสรุป (Sidebar Information)
with st.sidebar:
    st.header("📌 สรุปเงื่อนไขสำคัญ")
    st.write("- **วงเงินทุน:** ไม่เกิน 50,000 บาทต่อโครงการ [cite: 7, 8]")
    st.write("- **ระยะเวลาดำเนินการ:** ไม่เกิน 12 เดือน [cite: 6]")
    st.write("- **การจ่ายเงิน:** แบ่งจ่าย 3 งวด (50% / 30% / 20%) ")
    st.divider()
    if st.button("🗑️ ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.rerun()
