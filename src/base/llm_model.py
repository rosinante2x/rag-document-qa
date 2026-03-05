import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class LLM_Model:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY. Vui lòng kiểm tra file .env!")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview", # Hoặc model bạn đang dùng
            google_api_key=api_key,
            temperature=0.1
        )
        