import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.base.llm_model import get_llm
from src.rag.main import build_rag_chain, InputQA

llm = get_llm()
genai_docs = "./data_source/documents"

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

app = FastAPI(
    title="Vietnam Labor Law RAG System",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai")
async def generative_ai(inputs: InputQA):
    response = genai_chain.invoke({"question": inputs.question})

    answer_text = response.content if hasattr(response, "content") else str(response)

    return {"answer": answer_text}