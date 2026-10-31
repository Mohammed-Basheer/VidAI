import os
from operator import itemgetter

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm

# 1. Setup Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2. Initialize PGVector
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="video_audio_docs_kb",
    connection=os.environ.get("PGVECTOR_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db"),  # Note: requires psycopg3
    use_jsonb=True,
)

# 3. Define the LLM
llm = get_llm()

# 4. Create the RAG Prompt; answer directly first, only elaborate if it adds value
template = """Answer the user's question about the video transcript using the context below.
- Lead with a direct answer in the first sentence (e.g. "Yes, ...", "No, ...", or the core fact). Do not open with a restatement of the question.
- Only add brief supporting detail if it meaningfully helps; don't re-explain concepts you already covered in the conversation so far.
- If the answer is not in the context, say you do not know.

Conversation so far:
{chat_history}

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Helper function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str, video_id: str | None = None, chat_history: str = "(none yet)") -> str:
    # Scope retrieval to one ingested video so answers don't leak context from others
    search_kwargs = {"k": 8}
    if video_id is not None:
        search_kwargs["filter"] = {"video_id": video_id}
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke({"question": question, "chat_history": chat_history})
 