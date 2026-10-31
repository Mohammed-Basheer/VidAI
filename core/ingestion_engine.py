from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def ingest_transcript(transcript: str, video_title: str, vector_store, video_id: str):
    # 1. Split the raw transcript string into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    raw_chunks = text_splitter.split_text(transcript)

    # 2. Convert text chunks into LangChain Documents and attach metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={"source_type": "video", "title": video_title, "video_id": video_id},
        )
        for chunk in raw_chunks
    ]

    # 3. Insert into PostgreSQL
    vector_store.add_documents(documents)
    print(f"Successfully saved {len(documents)} video chunks to pgvector.")