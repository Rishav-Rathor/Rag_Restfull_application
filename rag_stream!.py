import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from pymilvus import connections


load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
ZILLIZ_URI = os.getenv('ZILLIZ_URI')
ZILLIZ_TOKEN = os.getenv('ZILLIZ_TOKEN')

st.title("RAG PDF Chatbot") 
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query = st.text_input("Ask a question about your PDF:")

if uploaded_file and user_query:
    
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PDFPlumberLoader("temp.pdf")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    connections.connect(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    vector_db = Milvus.from_documents(
        chunks,
        embedder,
        connection_args={
            "uri": ZILLIZ_URI,
            "token": ZILLIZ_TOKEN,
        },
        collection_name="rag_streamlit"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=512,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    answer = qa.invoke({"query": user_query})
    st.write("**Answer:**")
    st.write(answer['result'])
