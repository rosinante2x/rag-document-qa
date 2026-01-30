# from typing import Union
# from langchain_chroma import Chroma
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# class VectorDB:
#     def __init__(self,
#                  documents,
#                  vector_db: Union[Chroma, FAISS] = Chroma,
#                  embedding = HuggingFaceEmbeddings(),
#                  ) ->None:
        
#         self.vector_db = vector_db
#         self.embedding = embedding
#         self.db = self._build_db(documents)
        
#     def _build_db(self, documents):
#         db = self.vector_db.from_documents(documents=documents,
#                                            embedding=self.embedding)
#         return db
    
#     def get_retriever(self,
#                      search_type: str = "similarity",
#                      search_kwargs: dict = {"k": 10}
#                      ):
#         retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
#         return retriever

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
        self.embedding = embedding or HuggingFaceEmbeddings()
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        """Tạo vector DB, sinh id thủ công và truyền vào Chroma/FAISS"""
        ids = []
        fixed_docs = []

        for doc in documents:
            doc_id = str(uuid4())
            ids.append(doc_id)

            # Copy lại document, thêm id vào metadata
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**getattr(doc, "metadata", {}), "doc_id": doc_id},
            )
            fixed_docs.append(new_doc)

        # Truyền ids vào trực tiếp (Chroma hỗ trợ)
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
            search_kwargs = {"k": 10}
        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )


