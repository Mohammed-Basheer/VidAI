from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from core.llm import get_llm


# 1. Define your extraction schema using Pydantic
class ContentInsights(BaseModel):
    key_takeaways: List[str] = Field(description="The main points, lessons, or recommendations covered in the video.")
    topics_covered: List[str] = Field(description="The distinct topics or sections discussed, in order.")
    open_questions: List[str] = Field(description="Unresolved questions or topics the video leaves for further research.")


# 2. Initialize the LLM and bind the Pydantic schema to it
structured_llm = get_llm().with_structured_output(ContentInsights)

# 3. Create the extraction prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert content analyst. Extract the key takeaways, topics covered, and open questions from the following video transcript."),
    ("human", "{transcript}"),
])

# 4. Build the chain
extraction_chain = prompt | structured_llm


def extract_insights(transcript: str) -> ContentInsights:
    # A single API call returns a cleanly parsed Python object
    return extraction_chain.invoke({"transcript": transcript})
