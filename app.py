from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import pydub
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

from models import detect_audio

app = FastAPI(title="AI-Generated Voice Detection API - GUVI Hackathon")

API_KEY = os.getenv("API_KEY")


class AudioInput(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/api/voice-detection")
def detect_voice(
    input_data: AudioInput,
    x_api_key: str = Header(None, alias="x-api-key")
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if input_data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    try:
        audio_bytes = base64.b64decode(input_data.audioBase64)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segment = pydub.AudioSegment.from_file(tmp_path, format="mp3")
        finally:
            os.remove(tmp_path)

        segment = segment.set_channels(1).set_frame_rate(16000)
        samples = segment.get_array_of_samples()
        y = np.array(samples, dtype=np.float32) / 32768.0

        classification, confidence, explanation = detect_audio(y)

        return {
            "status": "success",
            "languageProvided": input_data.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root():
    return {"message": "AI Voice Detection API - Ready"}
