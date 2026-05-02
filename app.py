
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

# ---------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------
st.set_page_config(page_title="MLii Innovation Fund", page_icon="💡")
st.title("MLii Innovation Fund 💡")
st.markdown("Ask me anything about the information of Innovation fund!")

# ---------------------------------------------------------
# โหลด Model และข้อมูล (ใช้ Cache เพื่อไม่ให้โหลดใหม่ทุกครั้ง)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลด AI Model และข้อมูล... (อาจใช้เวลาหลายนาที)")
def setup_rag_system():
    file_path = "/content/extracted_ocr_data.txt"
    
    if not os.path.exists(file_path):
        st.error(f"ไม่พบไฟล์ข้อมูล '{file_path}' กรุณาอัปโหลดไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ app.py")
        return None

    # 1. Load and Split
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 2. Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. LLM (OpenThaiGPT 7B)
    model_id = "openthaigpt/openthaigpt-1.0.0-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 4. Prompt & Chain
    system_prompt = (
        "You are a knowledgeable and helpful assistant specialized in MFU Research Grants. "
        "Use the following pieces of retrieved context to answer the user's question about the grant rules and conditions. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Keep your answers concise, accurate, and helpful in Thai.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# โหลดระบบ RAG
rag_chain = setup_rag_system()

# ---------------------------------------------------------
# สร้าง UI สำหรับแชท
# ---------------------------------------------------------
if rag_chain:
    # เก็บประวัติการแชท
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ตัวอย่างคำถาม
    with st.expander("📝 ดูตัวอย่างคำถาม"):
        st.write("- อาจารย์จะตั้งงบวิจัยอย่างไร?")
        st.write("- ผู้วิจัยจะได้รับเงินเมื่อไหร่?")
        st.write("- ผู้วิจัยสามารถเบิกค่าตอบแทนได้หรือไม่?")

    # แสดงประวัติแชทเก่า
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # รับข้อความใหม่
    if user_query := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        # แสดงข้อความผู้ใช้
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # หาคำตอบ
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาคำตอบ..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
        
        # บันทึกคำตอบ
        st.session_state.messages.append({"role": "assistant", "content": answer})
