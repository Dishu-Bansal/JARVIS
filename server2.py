import signal, socket
import wave
import asyncio
from collections import deque
import numpy as np
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Inference, Model
import threading
from queue import Queue
import time, re, os, io, traceback, subprocess, tempfile
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
# from speechbrain.inference.speaker import SpeakerRecognition
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import soundfile as sf
from google import genai
from google.genai import types
from piper import PiperVoice
from fastapi import FastAPI, WebSocket
import uvicorn, torchaudio, clipboard, pyautogui, requests, json, wmi, pythoncom
from df.enhance import enhance, init_df, load_audio, save_audio
from VoiceTraining.testing import compare_wavs
from openai import OpenAI
from soniox_examples.speech_to_text.python.soniox_realtime import translate

app = FastAPI()

HOST = '0.0.0.0'
PORT = 50007

# ─────────────────────────────────────────────────────────────────────────
# Load Silero VAD model
# ─────────────────────────────────────────────────────────────────────────
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# ─────────────────────────────────────────────────────────────────────────
# Load Faster-Whisper with a smaller model for faster transcription
# ─────────────────────────────────────────────────────────────────────────
device = "cuda"           # or "cpu"
compute_type = "int8"     # Use int8 for faster inference
# model_whisper = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)

# ─────────────────────────────────────────────────────────────────────────
# Load Speaker Verification model and reference embedding
# ─────────────────────────────────────────────────────────────────────────
# model = Model.from_pretrained("pyannote/embedding", use_auth_token="<AUTH_TOKEN>")

# speaker_verification = Inference(model, window="whole").to(torch.device(device))

# reference_embedding_file = "reference_embedding.npy"
# if not os.path.exists(reference_embedding_file):
#     raise FileNotFoundError(f"Reference embedding file {reference_embedding_file} not found. Run enrollment script first.")
# reference_embedding = np.load(reference_embedding_file)
# VERIFICATION_THRESHOLD = 0.7  # Cosine similarity threshold for verification
# Load Speaker Embedding model and reference embedding
# try:
#     speaker_embedding = SpeakerRecognition.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb",
#         savedir="tmp_speechbrain",
#         run_opts={"device": device}
#     )
# except Exception as e:
#     raise RuntimeError(f"Failed to load speechbrain/spkrec-ecapa-voxceleb: {e}")
# reference_embedding_file = "reference_embedding.npy"
# if not os.path.exists(reference_embedding_file):
#     raise FileNotFoundError(f"Reference embedding file {reference_embedding_file} not found. Run enrollment script first.")
# reference_embedding = np.load(reference_embedding_file)
# print(f"Reference embedding shape: {reference_embedding.shape}, norm: {np.linalg.norm(reference_embedding):.3f}, first 5 values: {reference_embedding[:5]}")
# VERIFICATION_THRESHOLD = 0.7

encoder = VoiceEncoder()
# wav_enroll = preprocess_wav("dishu.wav")
# reference = embed_enroll = encoder.embed_utterance(wav_enroll)
reference = np.load("user_vec.npy")


# ─────────────────────────────────────────────────────────────────────────
# Setup Gemini
# ─────────────────────────────────────────────────────────────────────────
client = genai.Client(api_key="<YOUR_GEMINI_API_KEY>")
openai_client = OpenAI(
  api_key="<YOUR_OPENAI_API_KEY>"
)

def getCurrentText() -> str:
    """Fetches the text currently shown on the user’s screen. Use this to read what's visible.

    Args:
        None

    Returns:
        string - The currently selected text
    """
    return clipboard.paste()

def typeText(text: str) -> str:
    """Types in the specified text wherever the User's cursor is currently active

    Args:
        text: The text to type.

    Returns:
        A string telling if the typing was successfully executed or not.
    """
    try:
        pyautogui.typewrite(text)
        return "Typed Successfully"
    except Exception as e:
        return e

def WebSearchAgent(query: str) -> str:
    """A helper Web Search Agent. It will search the web for the query and returna natural language response.

    Args:
        query: The query to search.

    Returns:
        A string with the results.
    """
    client2 = genai.Client(api_key="<YOUR_GEMINI_API_KEY_2>")
    tools2 = [
    types.Tool(google_search=types.GoogleSearch()),
    ]
    response = client2.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[query],
                    config=types.GenerateContentConfig(tools=tools2, response_mime_type="text/plain", system_instruction="""You are specialized Web Search Agent. Your task is to search the web and return the best results for the query.""")
                )
    return response.text

def open_file_via_dockie(query: str) -> str:
    """
    Opens the best matching file based on the natural language query.

    Args:
        query (str): Natural language search query.

    Returns:
        str: Message indicating result of the action.
    """
    try:
        print(f"Opening file for: {query}")
        pythoncom.CoInitialize()
        c = wmi.WMI()
        system_info = c.Win32_ComputerSystemProduct()[0]
        id = system_info.UUID
        url = "https://dockie.codewithdishu.com/upload"
        payload = {
                'id': id,
                'text': query,
                'path': None
            }

        print("Payload: " + str(payload))
        response = requests.post(url, data=payload)
        response.raise_for_status()
        res = json.loads(response.text)
        print(res)

    except Exception as e:
        print(f"Error opening file: {e}")
        return f"Something went wrong: {e}"
    finally:
        print("Opening...")
        top_file = res['text']['1'][1]
        print(top_file)

        if not top_file:
            return "I couldn’t find any matching files."

        # top_file = results[0].get("file_path")
        if not top_file or not os.path.exists(top_file):
            return "The top matched file couldn’t be found on disk."
        
        os.startfile(top_file)
        return f"Opening the top match: {os.path.basename(top_file)}"

tools = [
    types.Tool(google_search=types.GoogleSearch()),
    ]
tools = [getCurrentText, typeText]
chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(tools=[getCurrentText, typeText, open_file_via_dockie, WebSearchAgent], response_mime_type="text/plain", system_instruction="""You are Jarvis, a voice-based AI assistant designed to interact with the user through natural conversation. Address the user as sir. Follow these instructions:
                                                                                        1) You are speaking, not typing. User is listening, not reading. So avoid reading out code or overly long answers. Summarise if needed or speak some ideas and ask user if you should continue the list.
                                                                                        2) Always keep the answers being spoken out loud short (just a few sentences). Details of the answer show on screen or ask user if you should elaborate.   
                                                                                        3) Use the following tags:
                                                                                            a) <speak></speak> - To speak something out loud. Kepp it concise, summarised, and short. Frequency - Always. Every response should have this tag because You are voice-based assistant.
                                                                                            b) <screen></screen> - To show the long form of the answer or anything on screen. Frequency - Only when needed. 
                                                                                            **IMPORTANT: Every part of the response should be covered in a tag**
                                                                                        4) When User says vague things like "What does this mean?" or "Analyze this", Check the on-screen text using the provided tool.
                                                                                        5) If User input is empty, or User suddenly goes out of context, or says gibberish, He is likely talking to someone else, not you. So just output empty tags <speak></speak> to mimic silence. Wait for User to get back to you.
                                                                                        6) Be proactive. You are friend. Get to know your user more. Ask questions, clarifications if needed. You can also ask questions just to get to know the user more."""))

piper_voice = PiperVoice.load(model_path="en_US-hfc_male-medium.onnx", config_path="en_US-hfc_male-medium.onnx.json")

# ─────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────
sample_rate = 16000
chunk_size = 512          # ~32ms
frame_ms = (chunk_size / sample_rate) * 1000.0
pre_roll_ms = 100
num_pre_roll_frames = int(pre_roll_ms // frame_ms)
MIN_UTTERANCE_SAMPLES = 8000  # ~0.5s for faster transcription
PAUSE_THRESHOLD_COMPLETE = 2000      # 300ms (9 chunks) for complete sentences
PAUSE_THRESHOLD_INCOMPLETE = 2000   # 800ms (25 chunks) for incomplete
latest_transcription = None  # Shared variable for latest transcription
# Queue for offloading transcription tasks
transcription_queue = Queue()
transcription_lock = threading.Lock()  # Lock for thread-safe access
lock = threading.Lock()
speaking = False
current_socket = None

def verify_from_bytearray(audio_bytes: bytearray, threshold=0.70):
    # Convert to numpy int16
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize to float32 [-1, 1]
    audio_float = audio_np.astype(np.float32) / 32768.0

    with wave.open("verify.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    # Write to memory buffer as WAV
    # wav_io = io.BytesIO()
    # sf.write(wav_io, audio_float, samplerate=16000, format='WAV')
    # wav_io.seek(0)

    # Preprocess and embed
    # wav = preprocess_wav("verify.wav")
    # test_vec = encoder.embed_utterance(wav)

    # sim = 1 - cosine(reference, test_vec)
    # print(f"Similarity: {sim:.3f}")
    sim = compare_wavs("D:/SPECIAL/Jarvis/voice_data/anchor/stitched_new_0.wav", "verify.wav")
    return sim >= threshold, sim

def buffer_to_waveform(audio_bytes: bytearray, sample_rate=16000):
    # Convert bytes → NumPy int16 → float32 [-1, 1]
    np_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(np_audio).unsqueeze(0)  # shape: (1, samples)
    return waveform, sample_rate

def extract_tags_in_order(text, tags):
    pattern = re.compile(r"(<speak>.*?</speak>|<screen>.*?</screen>|<type>.*?</type>)", re.DOTALL)
    matches = pattern.findall(text)

    extracted_data = []
    for match in matches:
        if "<speak>" in match:
            tag_type = "speak"
            content = re.search(r"<speak>(.*?)</speak>", match, re.DOTALL).group(1)
        elif "<screen>" in match:
            tag_type = "screen"
            content = re.search(r"<screen>(.*?)</screen>", match, re.DOTALL).group(1)
        elif "<type>" in match:
            tag_type = "type"
            content = re.search(r"<type>(.*?)</type>", match, re.DOTALL).group(1)
        else:
            continue
        extracted_data.append({"tag": tag_type, "content": content})
    return extracted_data
    
def gemini(query):
    response = chat.send_message(query)
    print(response.text)
    # match = re.search(r"<speak>(.*?)<\/speak>", response.text, re.DOTALL)
    # resp = match.group(1)
    # resp = str(resp).replace("*", "")
    return response.text

async def show(text):
    # print(f"[Screen]: {text}")
    time.sleep(3)  # Non-blocking wait

async def typeThis(text):
    # print(f"[Typing]: {text}")
    time.sleep(3)  # Non-blocking wait

async def speak(reply):
    global current_socket
    # Step 1: Synthesize audio using Piper
    # piper_voice.synthesize(reply, "voi.wav") 
    # raw_path = "voi.wav"

    # with wave.open(raw_path, 'wb') as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(2)
    #         piper_voice.synthesize(reply, wf)
    # # Step 2: Write raw audio to a temporary WAV file
    # # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw_wav_file:
    # #     raw_wav_file.write(raw_buffer.getvalue())
    # #     raw_wav_path = raw_wav_file.name
    # # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw_wav_file:
    # #     raw_wav_file.write(wav_data)
    # #     raw_path = raw_wav_file.name

    # with tempfile.NamedTemporaryFile(delete=False, suffix="_mobile.wav") as out_file:
    #     output_path = out_file.name

    # # Step 4: Use ffmpeg to normalize and resample
    # command = [
    #     "ffmpeg", "-y",
    #     "-i", raw_path,
    #     "-ac", "1",
    #     "-ar", "22050",         # match original sample rate
    #     "-filter:a", "loudnorm",
    #     output_path
    # ]

    # subprocess.run(command, check=True)

    # # Step 5: Read the processed audio and send to client
    # with open(output_path, "rb") as f:
    #     final_audio = f.read()

    # CHUNK_SIZE = 4096
    # for i in range(0, len(final_audio), CHUNK_SIZE):
    #     conn.sendall(final_audio[i:i+CHUNK_SIZE])
    # conn.sendall(b'__AUDIO_END__')
    # audio_buffer = io.BytesIO()
    # with lock:
    #     speaking = True
    for audio_bytes in piper_voice.synthesize_stream_raw(reply):
        await current_socket.send_bytes(audio_bytes)
    # with lock:
    #     speaking = False

async def stream_audio(text):
    for audio_bytes in piper_voice.synthesize_stream_raw(text):
        yield audio_bytes
        await asyncio.sleep(0)  # Yield control to event loop

def prepare_audio_for_mobile(input_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_mobile.wav"

    # Downsample, mono, normalize loudness, lower bitrate for smoother playback
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "24000",      # resample to 24kHz
        "-filter:a", "loudnorm",  # normalize loudness to -16 LUFS
        "-b:a", "96k",        # reduce bitrate
        output_path
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"✅ Processed audio saved to {output_path}")
    return output_path

# Create resamplers
resample_16_to_48 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
resample_48_to_16 = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
model, df_state, _ = init_df()

# Assume `input_audio` is a float32 numpy array at 16kHz
def denoise_byte_audio(audio_buffer: bytearray) -> torch.Tensor:
    # Step 1: Convert bytearray → float32 numpy [-1, 1]
    audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    # Step 2: Convert to tensor shape [C=1, T]
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]

    # Step 3: Resample to 48kHz
    audio_48k = resample_16_to_48(audio_tensor)  # [1, T]

    # Step 4: Pass through DeepFilterNet
    with torch.no_grad():
        denoised_48k = enhance(model, df_state, audio_48k)  # [1, T]

    # Step 5: Resample back to 16kHz
    denoised_16k = resample_48_to_16(denoised_48k)  # [1, T]

    final = denoised_16k.squeeze(0)  # shape: [T]
    int16_audio = (final.numpy() * 32768).astype(np.int16)
    return bytearray(int16_audio.tobytes())

async def transcription_worker():
    """Process transcription tasks from the queue and send results to client."""
    while True:
        task = transcription_queue.get()
        if task is None:  # Sentinel for shutdown
            break
        audio_buffer_noisy, utterance_count, is_verified, similarity = task
        utterance_filename = f"utterance_{utterance_count}_noisy.wav"
        with wave.open(utterance_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_buffer_noisy)
        audio_buffer = denoise_byte_audio(audio_buffer_noisy)
        if not is_verified:
            print(f"Utterance {utterance_count} rejected (similarity: {similarity:.3f})")
            # await speak("Sorry, I am not configured to serve you.")
            transcription_queue.task_done()
            continue
        else:
            print(f"Utterance {utterance_count} verified (similarity: {similarity:.3f})")

        utterance_filename = f"utterance_{utterance_count}.wav"
        with wave.open(utterance_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_buffer)

        # print(f"Transcribing {utterance_filename}...")
        start_time = time.time()
        # audio_data = read_audio(utterance_filename, sampling_rate=sample_rate)  # Use Silero's read_audio
        # segments, _ = model_whisper.transcribe(utterance_filename, task='translate', vad_filter=False)  # VAD handled by Silero
        text = translate(utterance_filename)
        print(f"text: {text}")
        # full_text = "".join(segment.text for segment in segments)
        # print(f"text: {full_text}")
        # print(f"Transcription took {time.time() - start_time:.3f}s")
        
        # resp = gemini(full_text)
        # tags = ["speak", "screen", "type"]
        # extracted_tags = extract_tags_in_order(resp, tags)
        # for item in extracted_tags:
        #     if item['tag'] == "speak":
        #         await speak(item["content"])
        #     if item['tag'] == "screen":
        #         await show(item["content"])
        #     if item['tag'] == "type":
        #         await typeThis(item["content"])

        # print("Response: " + response)
        # await speak(response)
        # Send transcription result to client
        # message_to_send = {'text': full_text}
        # try:
        #     # conn.send(str(message_to_send).encode())
        # except Exception as e:
        #     print(f"Error sending transcription: {e}")
        
        transcription_queue.task_done()

        # Check if transcription is a complete sentence
        # is_complete = bool(re.match(r'.*[.!?]$', full_text))
        # print(f"Transcription is {'complete' if is_complete else 'incomplete'}")

        # Update latest transcription and history
        # with transcription_lock:
        #     latest_transcription = {
        #         'text': full_text,
        #         'is_complete': is_complete,
        #         'timestamp': time.time()
        #     }
            # print("Latest: " + str(latest_transcription))
            # conversation_history.append({'role': 'user', 'content': full_text})

        # transcription_queue.task_done()

def run_transcription():
    asyncio.run(transcription_worker())

# def handle_client(conn, addr):
#     utterance_count = 0
#     proper_start_sent = False
#     vad_iterator = VADIterator(model_vad)
#     audio_buffer = bytearray()
#     triggered = False
#     ring_buffer = deque(maxlen=num_pre_roll_frames)
#     vad_buffer = np.array([], dtype=np.float32)
#     pause_chunks = 0  # Count non-speech chunks for pause detection
#     is_pausing = False  # Track if we're in a pause state

#     # Optimize socket for single client
#     # conn.settimeout(0.1)  # Short timeout for non-blocking behavior
#     conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)  # Larger buffer

#     while True:
#         try:
#             data = conn.recv(8192)  # Increased from 4096
#             if not data:
#                 break

#             pcm_samples = np.frombuffer(data, dtype=np.int16)
#             if pcm_samples.size == 0:
#                 continue
#             audio_float32 = pcm_samples.astype(np.float32) / 32768.0
#             vad_buffer = np.concatenate((vad_buffer, audio_float32))

#             while len(vad_buffer) >= chunk_size:
#                 current_chunk = vad_buffer[:chunk_size]
#                 vad_buffer = vad_buffer[chunk_size:]

#                 speech_segments = vad_iterator(current_chunk, return_seconds=False)
#                 is_speech_start = (speech_segments is not None and 'start' in speech_segments)
#                 is_speech_end = (speech_segments is not None and 'end' in speech_segments)

#                 chunk_int16 = (current_chunk * 32768.0).astype(np.int16).tobytes()

#                 if is_speech_start and not triggered:
#                     print(f"start detected at {speech_segments['start']}")
#                     for rb_chunk in ring_buffer:
#                         audio_buffer.extend(rb_chunk)
#                     ring_buffer.clear()
#                     triggered = True
#                     proper_start_sent = False
#                     is_pausing = False
#                     pause_chunks = 0

#                 if triggered:
#                     audio_buffer.extend(chunk_int16)
#                     if len(audio_buffer) >= MIN_UTTERANCE_SAMPLES and not proper_start_sent:
#                         print("Proper speech start detected (>=0.5s)")
#                         message_to_send = {'detection': 'proper_speech_start'}
#                         # conn.send(str(message_to_send).encode())
#                         proper_start_sent = True
#                 else:
#                     ring_buffer.append(chunk_int16)

#                 if is_speech_end and triggered:
#                     triggered = False
#                     if len(audio_buffer) >= MIN_UTTERANCE_SAMPLES:
#                         is_verified, similarity = verify_from_bytearray(audio_buffer)
#                         transcription_queue.put((audio_buffer, utterance_count, is_verified, similarity))
#                         utterance_count += 1
#                         is_pausing = True  # Start pause detection
#                         pause_chunks = 0
#                     else:
#                         message_to_send = {'detection': 'speech_false_detection'}
#                         # conn.send(str(message_to_send).encode())
#                     audio_buffer = bytearray()
#                     proper_start_sent = False
                
#                 # Pause detection
#                 if is_pausing and not is_speech_start:
#                     pause_chunks += 1
#                     pause_ms = pause_chunks * frame_ms
#                     with transcription_lock:
#                         if latest_transcription:
#                             threshold_ms = (
#                                 PAUSE_THRESHOLD_COMPLETE if latest_transcription['is_complete']
#                                 else PAUSE_THRESHOLD_INCOMPLETE
#                             )
#                             if pause_ms >= threshold_ms:
#                                 # Log final text for Gemini
#                                 # history = list(conversation_history)
#                                 print(f"Pause detected ({pause_ms:.0f}ms), final text for Gemini: {latest_transcription}")
#                                 is_pausing = False  # Reset pause detection
#                                 pause_chunks = 0
#                 elif is_speech_start:
#                     is_pausing = False
#                     pause_chunks = 0

#         except socket.timeout:
#             continue
#         except Exception as e:
#             traceback.print_exc()
#             print(f"Client error: {e}")
#             break

async def handle_client(conn: WebSocket, addr):
    utterance_count = 0
    vad_iterator = VADIterator(model_vad)
    audio_buffer = bytearray()
    triggered = False # Indicates if we are currently accumulating an utterance
    ring_buffer = deque(maxlen=num_pre_roll_frames) # Stores audio before speech starts
    vad_buffer = np.array([], dtype=np.float32)

    # Silence duration to trigger transcription (in milliseconds)
    SILENCE_DURATION_THRESHOLD_MS = 2000 
    
    # Store the timestamp of the last *speech activity* detected
    # This will be updated every time VAD says "speech"
    last_speech_activity_time = time.time() 

    # conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)

    while True:
        try:
            # data = conn.recv(8192)
            data = await conn.receive_bytes()
            # print("Received data: " + str(len(data)))
            with lock:
                if speaking:
                    continue
            if not data:
                break

            pcm_samples = np.frombuffer(data, dtype=np.int16)
            if pcm_samples.size == 0:
                continue
            audio_float32 = pcm_samples.astype(np.float32) / 32768.0
            vad_buffer = np.concatenate((vad_buffer, audio_float32))

            # Process vad_buffer in chunk_size increments
            while len(vad_buffer) >= chunk_size:
                current_chunk = vad_buffer[:chunk_size]
                vad_buffer = vad_buffer[chunk_size:]

                # Get VAD prediction for the current chunk
                # 0 = non-speech, 1 = speech
                speech_prob = model_vad(torch.from_numpy(current_chunk), sample_rate).item() 
                is_speech_in_chunk = speech_prob > 0.5 # Using a probability threshold

                chunk_int16 = (current_chunk * 32768.0).astype(np.int16).tobytes()

                if is_speech_in_chunk:
                    # If speech is detected, update the last speech activity time
                    last_speech_activity_time = time.time() 
                    if not triggered:
                        # Speech just started, clear ring buffer and start accumulating
                        print("Speech start detected. Starting utterance collection.")
                        await current_socket.send_text("user_speaking")
                        for rb_chunk in ring_buffer:
                            audio_buffer.extend(rb_chunk)
                        ring_buffer.clear()
                        triggered = True
                    
                    audio_buffer.extend(chunk_int16) # Always add speech chunks to buffer
                    
                else: # current_chunk is non-speech
                    if triggered:
                        # We are currently accumulating an utterance and this chunk is silence.
                        # Add this silence chunk to the buffer as part of the potential pause.
                        audio_buffer.extend(chunk_int16)
                        
                        # Check if the silence duration has exceeded the threshold since the *last* speech
                        silence_duration_ms = (time.time() - last_speech_activity_time) * 1000
                        
                        if silence_duration_ms >= SILENCE_DURATION_THRESHOLD_MS:
                            # Silence threshold met, end the utterance
                            if len(audio_buffer) >= MIN_UTTERANCE_SAMPLES:
                                print(f"Silence of {SILENCE_DURATION_THRESHOLD_MS}ms detected. Transcribing utterance {utterance_count}.")
                                is_verified, similarity = verify_from_bytearray(audio_buffer)
                                transcription_queue.put((audio_buffer, utterance_count, is_verified, similarity))
                                utterance_count += 1
                            else:
                                print("Silence detected, but audio buffer too short for transcription. Discarding.")
                            
                            # Reset for the next utterance
                            audio_buffer = bytearray()
                            triggered = False
                            ring_buffer.clear() # Clear ring buffer after processing utterance
                    else:
                        # Not triggered yet, just accumulate in ring buffer
                        ring_buffer.append(chunk_int16)
                        
                        # If not triggered and we've been accumulating non-speech in the ring buffer
                        # for a long time, we might want to clear it periodically to prevent
                        # it from holding too much irrelevant audio before actual speech.
                        # For now, maxlen handles this.

        except socket.timeout:
            # Handle cases where client stops sending data, but an utterance might be pending
            if triggered:
                silence_duration_ms = (time.time() - last_speech_activity_time) * 1000
                if silence_duration_ms >= SILENCE_DURATION_THRESHOLD_MS:
                    if len(audio_buffer) >= MIN_UTTERANCE_SAMPLES:
                        print(f"Silence of {SILENCE_DURATION_THRESHOLD_MS}ms detected (due to timeout). Transcribing utterance {utterance_count}.")
                        is_verified, similarity = verify_from_bytearray(audio_buffer)
                        transcription_queue.put((audio_buffer, utterance_count, is_verified, similarity))
                        utterance_count += 1
                    else:
                        print("Silence detected (timeout), but audio buffer too short. Discarding.")
                    audio_buffer = bytearray()
                    triggered = False
                    ring_buffer.clear()
            continue
        except Exception as e:
            traceback.print_exc()
            print(f"Client error: {e}")
            break

def shutdown_server(signum, frame):
    """Handle server shutdown."""
    global server_running, server_socket
    print("Shutting down server...")
    server_running = False
    transcription_queue.put(None)  # Signal worker to stop
    if server_socket:
        server_socket.close()

@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    global current_socket
    try:
        await websocket.accept()
        print("Client connected for audio")

        if current_socket and not current_socket.client_state.name == "DISCONNECTED":
            try:
                await current_socket.close(code=1001)
                print("Previous client disconnected.")
            except Exception as e:
                print(f"Error disconnecting previous client: {e}")

        # Assign new connection
        current_socket = websocket
        await handle_client(websocket, None)

    # try:
    #     while True:
    #         # Receive binary audio data from Flutter
    #         data = await websocket.receive_bytes()
    #         print(f"Received audio chunk of {len(data)} bytes")

    #         # Optional: process audio here
    #         # For now, just echo back
    #         await websocket.send_bytes(data)
    except Exception as e:
        print("Client disconnected:", e)

def start_server():
    global server_running, server_socket
    server_running = True
    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)

    # Start transcription worker thread, passing the client connection
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            server_socket = s
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            # s.settimeout(1)
            print(f"Server listening on {HOST}:{PORT}")

            while server_running:
                try:
                    conn, addr = s.accept()
                    print(f"Connected to {addr}")
                    # Start transcription worker for this client
                    threading.Thread(target=transcription_worker, args=(conn,), daemon=True).start()
                    handle_client(conn, addr)
                    conn.close()
                except socket.timeout:
                    continue
                except OSError:
                    break
                except Exception as e:
                    print(f"Accept error: {e}")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server stopped.")

if __name__ == '__main__':
    # start_server()
    threading.Thread(target=run_transcription, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=50007)