from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from pymilvus import connections
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Read credentials from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Connect to Zilliz/Milvus
try:
    connections.connect(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print(" Connected to Zilliz/Milvus")
except Exception as e:
    print(f" Failed to connect to Zilliz: {str(e)}")

# Health check route
@app.get("/ping")
def ping():
    return {"status": "OK"}

# Upload and process PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(" /upload endpoint hit")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # Save PDF
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        print(f" Saved: {temp_path}")

        # Load PDF
        loader = PDFPlumberLoader(temp_path)
        documents = loader.load()
        print(f" Loaded {len(documents)} document(s)")

        # Split
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        print(f" Split into {len(chunks)} chunks")

        # Embed
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("🔗 Generating embeddings...")

        # Store in vector DB
        Milvus.from_documents(
            chunks,
            embedder,
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN},
            collection_name="rag_pdf"
        )
        print("📦 Stored in vector database")

        os.remove(temp_path)
        print("🧹 Temp file deleted")

        return JSONResponse(content={"message": "PDF uploaded and indexed successfully."})

    except Exception as e:
        print(f" ERROR: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Ask a question on indexed PDF
@app.post("/query")
async def query_pdf(query: str = Body(..., embed=True)):
    print(f" /query hit with: {query}")
    try:
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Milvus(
            embedding_function=embedder,
            collection_name="rag_pdf",
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN}
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

        result = qa.invoke({"query": query})
        print(" Answer generated")
        return {"answer": result["result"]}

    except Exception as e:
        print(f" Query error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
