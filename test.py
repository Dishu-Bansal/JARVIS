# from pydub import AudioSegment
# import os
# import random

# def stitch_clips_to_5_to_7s(input_folder, output_folder, min_duration=5000, max_duration=8000, silence_ms=300):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     clips = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])
#     silence = AudioSegment.silent(duration=silence_ms)

#     current = AudioSegment.empty()
#     idx = 0

#     for clip_name in clips:
#         clip = AudioSegment.from_wav(os.path.join(input_folder, clip_name))

#         # If clip is longer than max_duration, split it
#         if len(clip) > max_duration:
#             start = 0
#             while start < len(clip):
#                 segment_length = random.randint(min_duration, max_duration)
#                 end = start + segment_length
#                 chunk = clip[start:end]

#                 if len(chunk) >= min_duration:
#                     chunk.export(os.path.join(output_folder, f"stitched_new_{idx}.wav"), format="wav")
#                     idx += 1

#                 start = end
#         else:
#             # Stitch shorter clips together
#             if len(current) + len(clip) + silence_ms <= max_duration:
#                 current += clip + silence
#             else:
#                 # If current is long enough, save it
#                 if len(current) >= min_duration:
#                     current.export(os.path.join(output_folder, f"stitched_new_{idx}.wav"), format="wav")
#                     idx += 1
#                     current = clip + silence
#                 else:
#                     # Pad current with silence if needed and save
#                     while len(current) < min_duration:
#                         current += silence
#                     current.export(os.path.join(output_folder, f"stitched_new_{idx}.wav"), format="wav")
#                     idx += 1
#                     current = clip + silence

#     # Save any final chunk if it meets minimum length
#     if len(current) >= min_duration:
#         current.export(os.path.join(output_folder, f"stitched_new_{idx}.wav"), format="wav")

#     print(f"✅ Created {idx + 1} stitched samples in '{output_folder}'.")

# # Example usage:
# stitch_clips_to_5_to_7s("D:/SPECIAL/Jarvis/otherVoice", "D:/SPECIAL/Jarvis/voice_data/negative")

from server2 import WebSearchAgent
print(WebSearchAgent("getmagical.com"))
# from datasets import load_dataset
# import torchaudio
# import torch
# import os
# from collections import defaultdict
# import random

# # Config
# SAVE_DIR = "D:/SPECIAL/Jarvis/otherVoice"
# os.makedirs(SAVE_DIR, exist_ok=True)
# TARGET_SR = 16000
# CLIPS_PER_SPEAKER = 5
# NUM_SPEAKERS = 20

# # Load VoxCeleb (just enough to cover our speaker needs)
# dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", trust_remote_code=True) # safe margin

# # Group by speaker_id
# speaker_to_clips = defaultdict(list)
# for row in dataset:
#     speaker = row["speaker_id"]
#     speaker_to_clips[speaker].append(row)

# # Pick N speakers
# selected_speakers = random.sample(list(speaker_to_clips.keys()), NUM_SPEAKERS)

# resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=TARGET_SR)

# count = 0
# for speaker_id in selected_speakers:
#     clips = random.sample(speaker_to_clips[speaker_id], min(CLIPS_PER_SPEAKER, len(speaker_to_clips[speaker_id])))

#     for i, row in enumerate(clips):
#         audio = row["audio"]
#         waveform = torch.tensor(audio["array"]).unsqueeze(0)

#         # Resample to 16 kHz if needed
#         if audio["sampling_rate"] != TARGET_SR:
#             waveform = torchaudio.functional.resample(waveform, orig_freq=audio["sampling_rate"], new_freq=TARGET_SR)

#         filename = f"{speaker_id}_clip{i+1}.wav"
#         filepath = os.path.join(SAVE_DIR, filename)

#         torchaudio.save(filepath, waveform, TARGET_SR)
#         print(f"Saved: {filename}")
#         count += 1

# print(f"\n✅ Done! Saved {count} .wav files across {NUM_SPEAKERS} speakers in {SAVE_DIR}")



# import os
# import torchaudio
# from pydub import AudioSegment

# # Input/output settings
# INPUT_DIR = "D:/hugging/datasets/downloads/extracted/b549fd6591f119197f1eb0e7307be0e0f65b4a60c21f8868fa8ce74ce4e9e49e/en_train_1"
# OUTPUT_DIR = "D:/SPECIAL/Jarvis/otherVoice"
# TARGET_SR = 16000

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# i=1
# # Convert and resample
# for root, _, files in os.walk(INPUT_DIR):
#     for file in files:
#         if not file.endswith(".mp3"):
#             continue

#         mp3_path = os.path.join(root, file)
#         relative_path = os.path.relpath(mp3_path, INPUT_DIR)
#         wav_path = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0] + ".wav")
#         os.makedirs(os.path.dirname(wav_path), exist_ok=True)

#         # Load MP3 using pydub
#         audio = AudioSegment.from_mp3(mp3_path)
#         audio = audio.set_channels(1)  # mono
#         audio = audio.set_frame_rate(TARGET_SR)

#         # Export as wav
#         audio.export(wav_path, format="wav")
#         print(f"Converted: {wav_path}")
#         if i == 100:
#             break



# from fastapi import FastAPI, WebSocket
# import uvicorn
# from piper import PiperVoice
# import numpy as np

# piper_voice = PiperVoice.load(model_path="en_US-hfc_male-medium.onnx", config_path="en_US-hfc_male-medium.onnx.json")

# app = FastAPI()

# # Define a route for the root URL ('/') that responds to GET requests.
# # When a GET request is made to this URL, the 'hello_world' function will be executed.
# @app.websocket("/ws/audio")
# async def audio_stream(websocket: WebSocket):
#     # try:
#     await websocket.accept()
#     print("Client connected for audio")
#     # threading.Thread(target=run_transcription, args=(websocket,), daemon=True).start()
#     # await handle_client(websocket, None)

#     try:
#         while True:
#             # Receive binary audio data from Flutter
#             data = await websocket.receive_bytes()
#             print(f"Received audio chunk of {len(data)} bytes")

#             # wav_buffer = io.BytesIO()
#             # with wave.open(wav_buffer, 'wb') as wf:
#             #     wf.setnchannels(1)
#             #     wf.setsampwidth(16 // 8)  # 16-bit = 2 bytes
#             #     wf.setframerate(22050)
#             #     wf.writeframes(b'')  # Empty frames to write header
#             # wav_buffer.seek(0)

#             # await websocket.send_bytes(wav_buffer.getvalue())
#             # piper_sample_rate = 22050  # Adjust based on Piper model (e.g., 22050 or 24000)
#             # target_sample_rate = 44100  # Match Android default
#             for audio_bytes in piper_voice.synthesize_stream_raw("Hello"):
#             #     # Convert bytes to numpy array (16-bit PCM)
#             #     # audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#             #     # # Resample to 44.1kHz
#             #     # resampled_audio = librosa.resample(audio_np, orig_sr=piper_sample_rate, target_sr=target_sample_rate)
#             #     # # Convert back to 16-bit PCM
#             #     # resampled_audio = (resampled_audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
#                 await websocket.send_bytes(audio_bytes)
#             # Optional: process audio here
#             # For now, just echo back
#             # await websocket.send_bytes(data)
#     except Exception as e:
#         print("Client disconnected:", e)

# # This block ensures the Flask development server runs only when the script is executed directly.
# if __name__ == '__main__':
#     # Get the port from an environment variable, or default to 5000.
#     # This is useful for deployment environments like Docker or PaaS.
#     # port = int(os.environ.get('PORT', 5000))
#     port = 50007

#     # Run the Flask application.
#     # debug=True enables debugging features (like reloader and debugger),
#     # which are helpful during development but should be False in production.
#     # host='0.0.0.0' makes the server accessible from any IP address,
#     # which is necessary if you're running it in a container or on a server
#     # that Cloudflare Tunnel needs to access.
#     uvicorn.run(app, host="0.0.0.0", port=50007)
# import clipboard
# print(clipboard.paste())