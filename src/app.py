import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "D:/huggingface_cache"
# os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes 

from src.base.llm_model import get_hf_llm 
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.2)
genai_docs = "./data_source/documents"

#----------Chain-----------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

#-----------App - FastAPI----------

app = FastAPI(
    title="Vietnam Labor Law RAG System",
    version=1.0,
    description="Legal-domain Retrieval-Augmented Generation system for Vietnamese Labor Law",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

#----------Routes - FastAPI-----------

@app.get("/check")
async def check():
    return {"status": "ok"}


@app.post("/generative_ai")
async def generative_ai(inputs: InputQA):
    docs = genai_chain.invoke(inputs.question)

    answer_text = docs.content if hasattr(docs, "content") else str(docs)

    return {
        "answer": answer_text,
        "note": "Citations are embedded inside the answer."
    }

#----------Langserve Routes - Playground--------
add_routes(app,
           genai_chain,
           #play_ground_type="default",
           path="/generative_ai")