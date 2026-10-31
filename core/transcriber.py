import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_chunk(chunk_path: str) -> str:
    with open(chunk_path, "rb") as audio_file:
        return _client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            language="en",
            response_format="text",
        )

def transcribe_chunks(chunks: list[str]) -> str:
    return " ".join(transcribe_chunk(chunk).strip() for chunk in chunks)