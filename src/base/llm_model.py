import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():

    os.environ["GOOGLE_API_KEY"] = "AIzaSyBud2k3nOgCCuP4iLfLjiZSswag0pxbElY"

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1
    )

    return llm