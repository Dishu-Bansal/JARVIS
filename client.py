import socket
import threading, time
import sounddevice as sd
import numpy as np
import queue

HOST = "127.0.0.1"  # Replace with your server IP
PORT = 50007
BUFFER_SIZE = 4096

# Queue for audio playback
playback_queue = queue.Queue()

def record_and_send(s):
    """Record mic and send audio to server continuously."""
    print("[CLIENT] Recording... Press Ctrl+C to stop.")
    samplerate = 16000
    chunk_duration = 0.1  # 100ms chunks for real-time streaming
    chunk_size = int(samplerate * chunk_duration)

    def callback(indata, frames, time, status):
        s.sendall(indata.tobytes())  # Send audio chunk immediately

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', blocksize=chunk_size, callback=callback):
            while True:
                time.sleep(0.1)  # Keep stream alive
    except KeyboardInterrupt:
        print("[CLIENT] Stopping recording...")
        s.sendall(b'__END__')  # Signal end of session

def receive_audio(s):
    """Receive TTS audio in chunks and queue for playback."""
    audio_data = b""
    while True:
        try:
            packet = s.recv(BUFFER_SIZE)
            if not packet:
                break
            if b"__AUDIO_END__" in packet:
                audio_data += packet.replace(b"__AUDIO_END__", b"")
                if audio_data:
                    playback_queue.put(audio_data)
                audio_data = b""
                continue
            audio_data += packet
        except ConnectionResetError:
            print("[CLIENT ERROR] Server closed the connection.")
            break

    playback_queue.put(audio_data)

def play_audio():
    """Play audio from queue."""
    try:
        while True:
            audio_data = playback_queue.get()
            if audio_data is None:
                break  # Exit signal
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            print("Length: " + str(len(audio_np)))

            sd.play(audio_np, samplerate=22050)
            sd.wait()
    except Exception as e:
        print(e)

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("[CLIENT] Connected to server")

    # Start playback thread
    playback_thread = threading.Thread(target=play_audio, daemon=True)
    playback_thread.start()

    # Start receiving thread for continuous responses
    receive_thread = threading.Thread(target=receive_audio, args=(s,), daemon=True)
    receive_thread.start()

    try:
        # Record and send audio continuously
        record_and_send(s)
    except Exception as e:
        print(f"[CLIENT ERROR] Recording error: {e}")
    finally:
        playback_queue.put(None)  # Stop playback thread
        playback_thread.join()
        s.close()
        print("[CLIENT] Done")

if __name__ == "__main__":
    main()
