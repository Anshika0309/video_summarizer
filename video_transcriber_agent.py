import os
import subprocess
import tempfile
from pytube import YouTube
import whisper
from transformers import pipeline

def download_youtube_video(url: str, output_path: str) -> str:
    yt = YouTube(url)
    stream = yt.streams.filter(only_video=False, file_extension='mp4').first()
    video_path = os.path.join(output_path, "video.mp4")
    stream.download(filename=video_path)
    return video_path

def extract_audio(video_path: str, audio_path: str) -> None:
    if os.path.exists(audio_path):
        os.remove(audio_path)
    command = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path, "-y"
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text: str, model_name: str = "facebook/bart-large-cnn") -> str:
    summarizer = pipeline("summarization", model=model_name)
    max_chunk_size = 1024  # tokens
    inputs = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summary = ""
    for chunk in inputs:
        summary_chunk = summarizer(chunk, max_length=180, min_length=30, do_sample=False)[0]['summary_text']
        summary += summary_chunk + " "
    return summary.strip()

def video_to_summary_from_link(video_url: str, model_size="base", summarizer_model="facebook/bart-large-cnn") -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        print("[INFO] Downloading video...")
        video_path = download_youtube_video(video_url, tmpdir)

        print("[INFO] Extracting audio...")
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)

        print("[INFO] Transcribing audio...")
        transcript = transcribe_audio(audio_path, model_size=model_size)

        print("[INFO] Summarizing transcript...")
        summary = summarize_text(transcript, summarizer_model)

        return summary

if __name__ == "__main__":
    url = input("Paste YouTube or video URL: ").strip()
    try:
        final_summary = video_to_summary_from_link(url)
        print("\n=== FINAL SUMMARY ===\n")
        print(final_summary)
    except Exception as e:
        print(f"❌ Error occurred: {e}")
