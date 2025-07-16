import os
import tempfile
import streamlit as st
from pytube import YouTube
import whisper
from transformers import pipeline
import subprocess

def download_youtube_video(url: str, output_path: str) -> str:
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    video_path = os.path.join(output_path, "video.mp4")
    stream.download(filename=video_path)
    return video_path

def extract_audio(video_path: str, audio_path: str) -> None:
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
    max_chunk_size = 1024  # max input token length
    inputs = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summary = ""
    for chunk in inputs:
        out = summarizer(chunk, max_length=180, min_length=30, do_sample=False)[0]
        summary += out["summary_text"] + " "
    return summary.strip()

def clean_youtube_url(url: str) -> str:
    """Remove extra YouTube query parameters like &pp=..."""
    if "&" in url:
        url = url.split("&")[0]
    return url


from urllib.parse import urlparse, parse_qs, urlunparse

def clean_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    video_id = query.get("v", [""])[0]
    if not video_id:
        return url
    return f"https://www.youtube.com/watch?v={video_id}"


# Streamlit UI
st.title("🎥 Video Transcriber & Summarizer")
st.write("Paste a YouTube link below, and I'll transcribe + summarize it for you.")

video_url = st.text_input("YouTube Video URL")

if st.button("Generate Summary"):
    if not video_url:
        st.warning("Please enter a valid YouTube URL.")
    else:
        video_url = clean_youtube_url(video_url) 
        with st.spinner("Downloading and processing..."):
            
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = download_youtube_video(video_url, tmpdir)
                    audio_path = os.path.join(tmpdir, "audio.wav")
                    extract_audio(video_path, audio_path)

                    transcript = transcribe_audio(audio_path)
                    summary = summarize_text(transcript)

                    st.success("✅ Done!")
                    st.subheader("🔤 Transcript:")
                    st.text_area("Transcript", transcript, height=200)
                    st.subheader("🧠 Summary:")
                    st.write(summary)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
