from typing import Union
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorDB:
    def __init__(
        self,
        documents,
        vector_db: Union[Chroma, FAISS] = Chroma,
        embedding=None,
    ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding or HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
)
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        ids = []
        fixed_docs = []

        for doc in documents:
            doc_id = str(uuid4())
            ids.append(doc_id)

            new_doc = Document(
                page_content=doc.page_content,
                metadata={**getattr(doc, "metadata", {}), "doc_id": doc_id},
            )
            fixed_docs.append(new_doc)

        if self.vector_db == Chroma:
            db = self.vector_db.from_documents(
                documents=fixed_docs,
                embedding=self.embedding,
                ids=ids,
                persist_directory="data_source/chroma_db",
            )
            db.persist()
        else:
            db = self.vector_db.from_documents(
                documents=fixed_docs,
                embedding=self.embedding,
                ids=ids,
            )

        return db

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict = None,
    ):

        if search_kwargs is None:
            search_kwargs = {
                "k": 5,
            }

        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

@classmethod
def load_existing(cls):
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory="data_source/chroma_db",
        embedding_function=embedding,
    )

    obj = cls.__new__(cls)
    obj.db = db
    obj.embedding = embedding
    obj.vector_db = Chroma
    return obj
