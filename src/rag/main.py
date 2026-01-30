from pydantic import BaseModel, Field, ConfigDict

from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG


# -------------------------------
# Input / Output Models
# -------------------------------
class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

    # ✅ Cho phép kiểu tùy ý (fix lỗi BaseMessage)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

    # ✅ Cho phép kiểu tùy ý
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -------------------------------
# RAG Chain Builder
# -------------------------------
def build_rag_chain(llm, data_dir, data_type):
    # Load documents
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)

    # Build retriever
    retriever = VectorDB(documents=doc_loaded).get_retriever()

    # Build final chain
    rag_chain = Offline_RAG(llm).get_chain(retriever)

    return rag_chain
