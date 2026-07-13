# from vosk import Model, KaldiRecognizer
# import pyaudio
# import json

# model = Model("vosk-model-en-us-0.22")  # or model-small-en-us-0.15
# rec = KaldiRecognizer(model, 16000)
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
# stream.start_stream()

# print("🎤 Speak into the mic...")

# while True:
#     data = stream.read(4000)
#     if rec.AcceptWaveform(data):
#         result = json.loads(rec.Result())
#         print(result['text'])
# import pyaudio
# import wave
# import numpy as np
# import torch
# from pyannote.audio import Inference
# import argparse
# import os

# CHANNELS = 1
# RATE = 16000
# audio_file = "dishu.wav"
# EMBEDDING_FILE = "reference_embedding.npy"

# from pyannote.audio import Model
# model = Model.from_pretrained("pyannote/embedding", 
#                               use_auth_token="<AUTH_TOKEN>")
# """Generate and save speaker embedding."""
# speaker_embedding = Inference(model, window="whole")

# # with wave.open(audio_file, 'rb') as wf:
# #     assert wf.getframerate() == RATE, "Audio must be 16 kHz"
# #     assert wf.getnchannels() == CHANNELS, "Audio must be mono"
# #     audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

# embedding = speaker_embedding(audio_file)
# # embedding = speaker_embedding({"waveform": torch.tensor(audio_data).unsqueeze(0), "sample_rate": RATE})
# np.save(EMBEDDING_FILE, embedding)
# print(f"Saved embedding to {EMBEDDING_FILE}")
# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
# import pyaudio
# import wave
# import numpy as np
# import torch
# from speechbrain.inference.speaker import SpeakerRecognition
# import argparse
# import os

# # Audio parameters
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 1024
# RECORD_SECONDS = 7
# OUTPUT_FILE = "dishu.wav"
# EMBEDDING_FILE = "reference_embedding.npy"

# """Generate and save speaker embedding."""
# try:
#     speaker_embedding = SpeakerRecognition.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb",
#         savedir="tmp_speechbrain",
#         run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
#     )
# except Exception as e:
#     raise RuntimeError(f"Failed to load speechbrain/spkrec-ecapa-voxceleb: {e}")

# with wave.open(OUTPUT_FILE, 'rb') as wf:
#     assert wf.getframerate() == RATE, "Audio must be 16 kHz"
#     assert wf.getnchannels() == CHANNELS, "Audio must be mono"
#     audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
#     audio_data = audio_data[np.abs(audio_data) > 0.01]
#     if len(audio_data) < RATE:
#         raise ValueError("Audio too short after trimming silence")

# waveform = torch.tensor(audio_data).unsqueeze(0)
# embedding = speaker_embedding.encode_batch(waveform).squeeze().cpu().numpy()
# print(f"Generated embedding with shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.3f}, first 5 values: {embedding[:5]}")
# np.save(EMBEDDING_FILE, embedding)
# print(f"Saved embedding to {EMBEDDING_FILE}")

from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

# === CONFIG ===
folder = Path("D:/SPECIAL/Jarvis/myVoice")   # Folder containing your audio files
output_path = "user_vec.npy"    # Where to save your enrolled vector

# === Load encoder ===
encoder = VoiceEncoder()

# === Process all .wav/.m4a/.mp3 files ===
embeddings = []
for audio_file in folder.glob("*"):
    if audio_file.suffix.lower() not in {".wav", ".m4a", ".mp3"}:
        continue
    print(f"📥 Processing: {audio_file.name}")
    wav = preprocess_wav(audio_file)
    embed = encoder.embed_utterance(wav)
    embeddings.append(embed)

# === Check results ===
if not embeddings:
    raise ValueError("No valid audio files found in the folder!")

# === Average embeddings to create final speaker vector ===
user_vec = np.mean(embeddings, axis=0)
np.save(output_path, user_vec)
print(f"✅ Enrolled speaker vector saved to: {output_path}")

