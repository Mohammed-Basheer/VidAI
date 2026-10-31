# VidAI

Turn a video (YouTube link or local audio/video file) into a searchable, chatty summary.

## What it does, in simple terms

1. **Take a video** — paste a YouTube URL or a local audio file path into the chat.
2. **Download & chunk the audio** — the video's audio is pulled down, converted to a clean format, and cut into smaller pieces (so it fits the transcription service's limits).
3. **Transcribe** — each audio chunk is sent to Groq's Whisper model and turned into text, then stitched back into one full transcript.
4. **Summarize** — the full transcript is sent to an LLM (Mistral) to generate a short title and a bullet-point summary.
5. **Extract insights** — a second pass over the transcript pulls out key takeaways, topics covered, and open questions.
6. **Store for Q&A** — the transcript is split into small text chunks and saved into a Postgres (pgvector) database, tagged to that specific video.
7. **Chat about it** — ask questions in the same chat; your question is matched against the stored chunks for *that video only*, and the LLM answers using just that context.

## Full application flow

```mermaid
sequenceDiagram
    actor U as User
    participant App as app.py
    participant Audio as audio_processor.py
    participant Groq as Groq Whisper API
    participant Trans as transcriber.py
    participant Mistral as Mistral API
    participant Summ as summarise.py
    participant Extr as extractor.py
    participant Ing as ingestion_engine.py
    participant VS as vector_store.py
    participant PG as Postgres (pgvector)

    U->>App: opens chat
    App-->>U: welcome message (on_chat_start)

    U->>App: pastes YouTube URL / file path
    activate App
    App->>Audio: process_input(source)
    Audio-->>App: chunks (wav paths)

    App->>Trans: transcribe_chunks(chunks)
    loop each chunk
        Trans->>Groq: transcribe audio chunk
        Groq-->>Trans: chunk text
    end
    Trans-->>App: transcript (joined string)

    App->>Summ: generate_summary(transcript)
    Summ->>Mistral: invoke summary_chain
    Mistral-->>Summ: summary
    Summ-->>App: summary

    App->>Summ: generate_title(summary)
    Summ->>Mistral: invoke title_chain
    Mistral-->>Summ: title
    Summ-->>App: title

    App->>Extr: extract_insights(transcript)
    Extr->>Mistral: invoke extraction_chain
    Mistral-->>Extr: ContentInsights
    Extr-->>App: insights

    App->>Ing: ingest_transcript(transcript, title, vector_store, video_id)
    Ing->>VS: add_documents(chunked Documents)
    VS->>VS: embed each chunk (HuggingFace BGE)
    VS->>PG: INSERT INTO langchain_pg_embedding
    PG-->>VS: ok
    VS-->>Ing: ok
    Ing-->>App: ok

    App->>Audio: cleanup_files(chunks)
    Audio-->>App: temp files deleted

    App-->>U: "Processed <title>" + [Show Summary] [Show Key Takeaways] buttons
    deactivate App

    alt clicks "Show Summary"
        U->>App: show_summary action
        App-->>U: title + summary (read from session)
    else clicks "Show Key Takeaways & Topics"
        U->>App: show_insights action
        App-->>U: takeaways / topics / open questions (read from session)
    else asks a question
        U->>App: types question
        App->>VS: ask_question(question, video_id, chat_history)
        VS->>VS: embed_query(question)
        VS->>PG: SELECT ... WHERE video_id = ? ORDER BY embedding <=> query LIMIT 8
        PG-->>VS: top 8 matching chunks
        VS->>Mistral: invoke rag_chain(context, question, chat_history)
        Mistral-->>VS: answer
        VS-->>App: answer
        App-->>U: answer
        App->>App: append (question, answer) to chat_history
    end
```

## Running it

```powershell
chainlit run app.py -w
```

Opens a chat UI at `http://localhost:8000`. Paste a YouTube URL or local file path to process a video, then ask questions about it.

## Requirements

- Python deps: `pip install -r requirements.txt`
- A running Postgres instance with the `pgvector` extension (see `docker run ... pgvector/pgvector:pg16`)
- A `.env` file with `MISTRAL_API_KEY`, `GROQ_API_KEY`, and `PGVECTOR_URL` (your Postgres connection string)

## Project layout

- `app.py` — the Chainlit chat UI and orchestration
- `utils/audio_processor.py` — download, convert, and chunk audio
- `core/transcriber.py` — audio chunks → text (Groq Whisper)
- `core/summarise.py` — transcript → title + summary (Mistral)
- `core/extractor.py` — transcript → key takeaways / topics / open questions (Mistral)
- `core/ingestion_engine.py` — transcript → chunks stored in pgvector
- `core/vector_store.py` — retrieval + question answering (RAG) over stored chunks
- `core/llm.py` — shared Mistral client setup
