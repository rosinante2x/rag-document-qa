import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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
        Bạn là một chuyên gia pháp lý tại Việt Nam.

        Dựa vào các ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng.
        YÊU CẦU BẮT BUỘC VỀ ĐỊNH DẠNG:
        - Luôn luôn bắt đầu câu trả lời bằng cấu trúc: "Theo [Tên Đạo Luật] [Điều X], ..." dựa trên phần [Nguồn: ...] có trong ngữ cảnh.
        - Trả lời ngắn gọn, súc tích và chính xác.

        Nếu không có thông tin trong ngữ cảnh, chỉ trả lời:
        "Không tìm thấy thông tin trong tài liệu."

        Ngữ cảnh:
        {context}

        Câu hỏi:
        {question}

        Trả lời:
        """

        self.prompt = PromptTemplate.from_template(template)
        self.str_parser = Str_OutputParser()
        
    def get_chain(self, retriever):

        rag_chain = (
            {
                "context": RunnableLambda(
                    lambda x: self.format_docs(
                        retriever.invoke(x["question"])
                    )
                ),
                "question": RunnableLambda(lambda x: x["question"]),
            }
            | self.prompt
            | self.llm
            | self.str_parser
        )

        return rag_chain
    
    def format_docs(self, docs, max_chars=3000):
        domain_map = {
            "luat_lao_dong": "Bộ luật Lao động 2019",
            "luat_hon_nhan_gia_dinh": "Luật Hôn nhân và gia đình 2014",
            "luat_giao_thong": "Luật Giao thông đường bộ 2008",
            "luat_dan_su": "Bộ luật Dân sự 2015",
            "luat_bhxh": "Luật Bảo hiểm xã hội 2024"
        }
        
        formatted = []
        total_chars = 0

        for doc in docs:
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                break

            dieu = doc.metadata.get("dieu", "N/A")
            domain_code = doc.metadata.get("domain", "")
            law_name = domain_map.get(domain_code, "Tài liệu pháp luật")
            formatted.append(f"[Nguồn: {law_name}, Điều {dieu}]\n{content}")
            total_chars += len(content)

        return "\n\n".join(formatted)