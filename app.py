import asyncio
import uuid

import chainlit as cl

from utils.audio_processor import process_input, cleanup_files, YOUTUBE_URL_PREFIX
from core.transcriber import transcribe_chunks
from core.summarise import generate_summary, generate_title
from core.extractor import extract_insights
from core.ingestion_engine import ingest_transcript
from core.vector_store import ask_question, vector_store


MAX_HISTORY_TURNS = 6

# Session keys, centralized to avoid typos across handlers
SESSION_INGESTED = "ingested"
SESSION_VIDEO_ID = "video_id"
SESSION_CHAT_HISTORY = "chat_history"
SESSION_TITLE = "title"
SESSION_SUMMARY = "summary"
SESSION_INSIGHTS = "insights"
SESSION_PENDING_FILES = "pending_files"  # temp files for the in-progress video, if Cancel is clicked mid-run


def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none yet)"
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-MAX_HISTORY_TURNS:])


def format_bullets(label: str, items: list[str]) -> str:
    return f"**{label}**\n" + "\n".join(f"- {i}" for i in items)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set(SESSION_INGESTED, False)
    cl.user_session.set(SESSION_VIDEO_ID, None)
    cl.user_session.set(SESSION_CHAT_HISTORY, [])
    cl.user_session.set(SESSION_PENDING_FILES, None)
    await cl.Message(
        content="Paste a YouTube URL or local audio file path to process a video. "
        "Once it's ingested, ask questions about it here."
    ).send()


@cl.on_stop
async def on_stop() -> None:
    # The asyncio task behind process_source() is cancelled here, but any blocking
    # work already dispatched to a worker thread (download/transcribe/summarize calls)
    # keeps running in the background regardless - we can only clean up local files and state.
    pending_files = cl.user_session.get(SESSION_PENDING_FILES)
    if pending_files:
        cleanup_files(pending_files)
    cl.user_session.set(SESSION_PENDING_FILES, None)
    cl.user_session.set(SESSION_INGESTED, False)
    cl.user_session.set(SESSION_VIDEO_ID, None)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()

    if text.startswith(YOUTUBE_URL_PREFIX) or text.lower().endswith((".wav", ".mp3", ".m4a")):
        await process_source(text)
        return

    if not cl.user_session.get(SESSION_INGESTED):
        await cl.Message(content="No video has been ingested yet. Paste a YouTube URL or audio file path first.").send()
        return

    video_id = cl.user_session.get(SESSION_VIDEO_ID)
    history = cl.user_session.get(SESSION_CHAT_HISTORY)
    answer = await asyncio.to_thread(ask_question, text, video_id, format_history(history))

    history.append((text, answer))
    cl.user_session.set(SESSION_CHAT_HISTORY, history)
    await cl.Message(content=answer).send()


async def process_source(source: str) -> None:
    video_id = str(uuid.uuid4())

    async with cl.Step(name="Processing video") as step:
        step.output = "Downloading/chunking audio..."
        chunks = await asyncio.to_thread(process_input, source)
        cl.user_session.set(SESSION_PENDING_FILES, chunks)  # so on_stop can clean these up if cancelled now

        step.output = "Transcribing..."
        transcript = await asyncio.to_thread(transcribe_chunks, chunks)

        step.output = "Summarizing..."
        summary = await asyncio.to_thread(generate_summary, transcript)
        title = await asyncio.to_thread(generate_title, summary)
        insights = await asyncio.to_thread(extract_insights, transcript)

        step.output = "Saving to vector store..."
        await asyncio.to_thread(ingest_transcript, transcript, title, vector_store, video_id)

        step.output = "Cleaning up temporary audio files..."
        await asyncio.to_thread(cleanup_files, chunks)

    cl.user_session.set(SESSION_PENDING_FILES, None)
    cl.user_session.set(SESSION_INGESTED, True)
    cl.user_session.set(SESSION_VIDEO_ID, video_id)
    cl.user_session.set(SESSION_CHAT_HISTORY, [])  # new video, so drop Q&A history from any previous one
    cl.user_session.set(SESSION_TITLE, title)
    cl.user_session.set(SESSION_SUMMARY, summary)
    cl.user_session.set(SESSION_INSIGHTS, insights)

    await cl.Message(
        content=f"Processed **{title}**. Pick an option below, or just ask a question about the video.",
        actions=[
            cl.Action(name="show_summary", payload={}, label="Show Summary"),
            cl.Action(name="show_insights", payload={}, label="Show Key Takeaways & Topics"),
        ],
    ).send()


@cl.action_callback("show_summary")
async def on_show_summary(action: cl.Action) -> None:
    title = cl.user_session.get(SESSION_TITLE)
    summary = cl.user_session.get(SESSION_SUMMARY)
    await cl.Message(content=f"## {title}\n\n{summary}").send()


@cl.action_callback("show_insights")
async def on_show_insights(action: cl.Action) -> None:
    insights = cl.user_session.get(SESSION_INSIGHTS)
    insights_text = "\n\n".join([
        format_bullets("Key takeaways", insights.key_takeaways),
        format_bullets("Topics covered", insights.topics_covered),
        format_bullets("Open questions", insights.open_questions),
    ])

    await cl.Message(content=insights_text).send()

