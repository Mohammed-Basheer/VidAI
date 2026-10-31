import yt_dlp
import os
from pydub import AudioSegment
from imageio_ffmpeg import get_ffmpeg_exe
FFMPEG_EXE = get_ffmpeg_exe()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

YOUTUBE_URL_PREFIX = "https://www.youtube.com/"

# imageio-ffmpeg installs the binary inside site-packages, so it isn't on PATH for yt-dlp/pydub to find
AudioSegment.converter = FFMPEG_EXE
AudioSegment.ffmpeg = FFMPEG_EXE

def download_youtube_audio(url: str) -> str:
    output_path= os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'ffmpeg_location': FFMPEG_EXE,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        "quiet": True  # Suppress logging from yt_dlp
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url, download=True)
        file_name=ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return file_name

def convert_to_wav(file_path: str) -> str:
    wav_path = os.path.splitext(file_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    return wav_path

def chunk_audio(wav_path: str, chunk_minutes: int = 7) -> list[str]:
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000
    chunks = []
    base_path = os.path.splitext(wav_path)[0]
    for chunk_num, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_path = f"{base_path}_chunk_{chunk_num}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(url: str) -> list[str]:
    downloaded_path = None

    if url.startswith(YOUTUBE_URL_PREFIX):
        print("Downloading YouTube audio...")
        downloaded_path = download_youtube_audio(url)
        wav_path = convert_to_wav(downloaded_path)
    else:
        print("Detected local audio file...")
        wav_path = convert_to_wav(url)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Generated {len(chunks)} chunks.")

    # The downloaded/converted files are fully superseded by the chunks now, never touches the user's original local file
    cleanup_files([p for p in (downloaded_path, wav_path) if p])

    return chunks


def cleanup_files(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
