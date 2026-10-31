from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm

_llm = get_llm()

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an executive assistant. Provide a concise, structured summary of the following meeting/video transcript using bullet points and clear thematic headers."),
    ("human", "{transcript}"),
])
summary_chain = summary_prompt | _llm | StrOutputParser()

title_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a concise and descriptive title for the following meeting summary."),
    ("human", "{text}"),
])
title_chain = title_prompt | _llm | StrOutputParser()


def generate_summary(transcript: str) -> str:
    return summary_chain.invoke({"transcript": transcript})


def generate_title(summary: str) -> str:
    # Titles only need the opening of the summary, not the full text
    return title_chain.invoke({"text": summary[:3000]})
