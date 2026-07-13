import socket
import threading
import io, sys, signal
import whisper
import sounddevice as sd
from pyannote.audio import Pipeline
from piper import PiperVoice
import numpy as np
import soundfile as sf
import tempfile
import torch, json
import torchaudio
from vosk import Model, KaldiRecognizer
from silero_vad.utils_vad import VADIterator
from silero_vad import load_silero_vad
import nltk

print("CUDA: " + str(torch.cuda.is_available()))

# Initialize models
# asr_model = whisper.load_model("base", device=torch.device("cuda"), in_memory=True)
# diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="<AUTH_TOKEN>")
# diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
piper_voice = PiperVoice.load(model_path="en_US-hfc_male-medium.onnx", config_path="en_US-hfc_male-medium.onnx.json")

# Socket server setup
HOST = '0.0.0.0'
PORT = 50007
BUFFER_SIZE = 4096
SAMPLERATE = 16000
CHUNK_DURATION = 0.03  # 30ms chunks for VAD
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)

# Global flag for shutdown
server_running = True
server_socket = None  # Will hold the listening socket

model = Model("vosk-model-en-us-0.22")  # or model-small-en-us-0.15
recognizer = KaldiRecognizer(model, 16000)
recognizer.SetWords(True)

def is_complete_sentence(text):
    """Check if text forms a complete sentence or paragraph using NLTK."""
    sentences = nltk.sent_tokenize(text.strip())
    if not sentences:
        return False
    # Consider text complete if it ends with a sentence-ending punctuation
    return text.strip()[-1] in '.!?' and len(sentences) >= 1

def process_audio(audio_buffer, conn, text_buffer):
    """Process audio buffer: transcribe, append to text buffer, send complete sentences to LLM."""
    data = audio_buffer.getvalue()
    audio_array = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
    if len(audio_array) == 0:
        return text_buffer

    # # Save temporary WAV file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
    #     sf.write(temp_wav.name, audio_array, samplerate=SAMPLERATE, format="WAV", subtype="PCM_16")
    #     audio_path = temp_wav.name

    # print("File: " + audio_path)
    transcription = None
    with torch.inference_mode():
        # Speaker diarization
        # diarization = diarization_pipeline(audio_path)
        # print("Diarization: " + str(diarization))

        # Whisper transcription
        # transcription = asr_model.transcribe(audio_path)
        # Feed raw PCM to Vosk
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            transcription = text
            if text:
                print(f"Object: {str(result)}")
                print(f"Transcript: {text}")
        else:
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            if partial:
                transcription = partial
                print(f"Partial: {partial}")
    # if transcription['language'] != 'en':
    if transcription is None:
        return text_buffer
    text = transcription.strip()
    print(f"Transcription: {text}")
    if text:
        text_buffer.append(text)

    # Check if text buffer forms complete sentences
    full_text = " ".join(text_buffer)
    if is_complete_sentence(full_text):
        print(f"Complete text for LLM: {full_text}")

        # Generate response (replace with actual LLM call)
        reply = "It's working!"
        print(f"Response: {reply}")

        # TTS with Piper
        audio_buffer = io.BytesIO()
        for audio_bytes in piper_voice.synthesize_stream_raw(reply):
            audio_buffer.write(audio_bytes)

        # Send TTS audio
        CHUNK_SIZE = 4096
        audio_data = audio_buffer.getvalue()
        for i in range(0, len(audio_data), CHUNK_SIZE):
            conn.sendall(audio_data[i:i+CHUNK_SIZE])
        conn.sendall(b'__AUDIO_END__')

        # Clear text buffer after processing
        return []
    return text_buffer

def shutdown_server(signum, frame):
    """Handle server shutdown."""
    global server_running, server_socket
    print("Shutting down server...")
    server_running = False
    if server_socket:
        server_socket.close()

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    audio_buffer = io.BytesIO()
    text_buffer = []  # Buffer for accumulating transcriptions

    def init_vad():
        raw_model = load_silero_vad()
        vad = VADIterator(
            model=raw_model,
            sampling_rate=16000,
            threshold=0.3,
            # min_silence_duration_ms=500,
            # speech_pad_ms=100
        )
        try:
            vad.model = torch.jit.script(vad.model.eval().to("cuda"))
        except Exception:
            pass
        return vad
    # Initialize Silero VAD
    vad_iterator = init_vad()

    try:
        while server_running:
            data = conn.recv(BUFFER_SIZE)
            CHUNK_SIZE = 512
            if not data:
                break
            if data == b'__END__':
                if audio_buffer.getvalue():  # Process any remaining audio
                    text_buffer = process_audio(audio_buffer, conn, text_buffer)
                if text_buffer:  # Process any remaining text
                    full_text = " ".join(text_buffer)
                    if full_text.strip():
                        print(f"Final text for LLM: {full_text}")
                        reply = "It's working!"
                        audio_buffer = io.BytesIO()
                        for audio_bytes in piper_voice.synthesize_stream_raw(reply):
                            audio_buffer.write(audio_bytes)
                        
                        audio_data = audio_buffer.getvalue()
                        for i in range(0, len(audio_data), CHUNK_SIZE):
                            conn.sendall(audio_data[i:i+CHUNK_SIZE])
                        conn.sendall(b'__AUDIO_END__')
                break

            # Append data to audio buffer
            # audio_buffer.write(data)

            # Process audio in chunks for VAD
            # audio_array = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
            audio_array = np.frombuffer(data, dtype=np.int16)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                sf.write(temp_wav.name, audio_array, samplerate=SAMPLERATE, format="WAV", subtype="PCM_16")
                print("Audio:" + temp_wav.name)
            # print(len(audio_array))
            for i in range(0, len(audio_array), CHUNK_SIZE):
                chunk = audio_array[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    # print("Incomplete Chunk")
                    continue  # Incomplete chunk, wait for more data
                
                # print(f"[SERVER] Processing chunk, length: {len(chunk)}, max amplitude: {np.max(np.abs(chunk))}")
                # Convert chunk to format expected by Silero VAD (float32, normalized)
                chunk_float = chunk.astype(np.float32) / 32768.0
                # Process chunk with VAD
                with torch.inference_mode():
                    vad_output = vad_iterator(torch.from_numpy(chunk_float).to(torch.device("cuda")), return_seconds=False)
                # print("VAD: " + str(vad_output))
                if vad_output:
                    print("Adding data...")
                    audio_buffer.write(data)
                if vad_output and 'end' in vad_output:  # Speech segment ended (pause detected)
                    print("Prcessing...")
                    text_buffer = process_audio(audio_buffer, conn, text_buffer)
                    audio_buffer = io.BytesIO()  # Reset audio buffer
                    vad_iterator.reset_states()  # Reset VAD state

        vad_iterator.reset_states()  # Clean up
        conn.close()
    except Exception as e:
        print(f"Client error: {e.__traceback__}")
    finally:
        conn.close()
        print(f"Disconnected {addr}")

def start_server():
    global server_running, server_socket
    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            server_socket = s
            s.bind((HOST, PORT))
            s.listen()
            s.settimeout(1)  # Set timeout before accept loop
            print(f"Server listening on {HOST}:{PORT}")

            while server_running:
                try:
                    conn, addr = s.accept()
                    threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed or other error
                    break
                except Exception as e:
                    print(f"Accept error: {e}")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server stopped.")


if __name__ == '__main__':
    start_server()