import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()


def get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="ministral-3b-2512",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )
