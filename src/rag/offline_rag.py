import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
        
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response
        
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        template = """
        Bạn là chuyên gia pháp lý về Luật Lao động Việt Nam.

        Chỉ được trả lời dựa trên nội dung trong phần NGỮ CẢNH.
        Nếu không tìm thấy thông tin, trả lời đúng câu sau:
        "Không tìm thấy thông tin trong Luật Lao động hiện hành."

        Yêu cầu:
        - Trả lời rõ ràng, ngắn gọn
        - Phải trích dẫn Điều luật nếu có
        - Không tự suy luận ngoài tài liệu

        NGỮ CẢNH:
        {context}

        CÂU HỎI:
        {question}

        Answer:
        """

        self.prompt = PromptTemplate.from_template(template)
        self.str_parser = Str_OutputParser()
        
    def get_chain(self, retriever):

        def get_context(question):
            docs = retriever.invoke(question)
            return docs

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x),
                question=lambda x: x
            )
            | self.prompt
            | self.llm
        )

        return rag_chain
    
    def format_docs(self, docs):
        formatted = []
        for doc in docs:
            dieu = doc.metadata.get("dieu", "N/A")
            formatted.append(f"[Điều {dieu}]\n{doc.page_content}")
        return "\n\n".join(formatted)