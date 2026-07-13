import sounddevice as sd
from piper.voice import PiperVoice
from google import genai
from dotenv import load_dotenv 
from whisper_live.client import TranscriptionClient
import os, time
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1")
WHISPER_SERVER_URL = "http://localhost:9090"
MODEL_PATH = "en_US-hfc_male-medium.onnx"
CONFIG_PATH = "en_US-hfc_male-medium.onnx.json"

# ─── SETUP GEMINI ─────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)
model = client.models

# ─── SETUP PIPER ──────────────────────────────────────────

voice = PiperVoice.load(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH
)

def speak(text):
    # Setup a sounddevice OutputStream with appropriate parameters
    # The sample rate and channels should match the properties of the PCM data
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()

    for audio_bytes in voice.synthesize_stream_raw(text):
        int_data = np.frombuffer(audio_bytes, dtype=np.int16)
        stream.write(int_data)

    time.sleep(0.2)
    stream.stop()
    stream.close()


# ─── CALLBACK FOR WHISPER ─────────────────────────────────
last_update = time.time()
def on_transcription(text: str):
    # if not text.strip():
    #     return

    print(f"\nUser: {text}")
    response = model.generate_content(model="gemini-1.5-flash-latest",  contents=[text]).text
    print(f"Gemini: {response}")
    speak(response)

def on_data(data, data2):
    print("Transcription:" + str(data) + ";;" + str(data2))
# ─── START CLIENT ─────────────────────────────────────────

client = TranscriptionClient(
    host='localhost',
    port=9096,
    transcription_callback=on_data,
    lang="en"  # Only English
)

client()

print("Not Blocking")