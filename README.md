# video_summarizer
A simple yet powerful Streamlit app that lets you:

Paste a YouTube video link

Automatically download the audio

Transcribe the speech to text using OpenAI's whisper

Summarize the transcription using a Transformer-based summarizer

Video Transcribing Agent

A simple yet powerful **Streamlit app** that lets you:
- Paste a **YouTube video link**
- Automatically **download the audio**
- **Transcribe** the speech to text using OpenAI's `whisper`
- Summarize the transcription using a **Transformer-based summarizer**

---

Features

Paste a YouTube video link — no download needed on your end
- Automatic audio extraction via `pytube` + `ffmpeg`
- Transcription using `whisper` (OpenAI)
- Summarization using HuggingFace Transformers (e.g., `facebook/bart-large-cnn`)
- Clean UI with **Streamlit** for ease of use
- Works locally on most videos

---

## 🧠 Tech Stack

Video Download | `pytube`                     
Audio Extraction | `ffmpeg` via `subprocess`     
Transcription   | `whisper` (OpenAI)             
Summarization   | `transformers` from HuggingFace 
UI             | `Streamlit`                  
Language       | Python 3.9+                  

---

Installation

Clone this repository
git clone https://github.com/your-username/video-transcriber-agent.git
cd video-transcriber-agent


Create a virtual env
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For macOS/Linux

Run the app
streamlit run app.py

