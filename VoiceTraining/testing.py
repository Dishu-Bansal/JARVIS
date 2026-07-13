import torch
import torchaudio
import sys
from VoiceTraining.TripleNet import TripletNet
from pyannote.audio import Inference
import os

# --- Config ---
MODEL_PATH = "D:/SPECIAL/Jarvis/tripletnet_dishu.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Trained Model ---
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = TripletNet(input_dim=checkpoint["input_dim"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Load ECAPA embedding extractor ---
spkrec = Inference("pyannote/embedding", window="whole", use_auth_token="<AUTH_TOKEN>").to(device)

# --- Audio Preprocessing ---
def preprocess_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform[:1]
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.to(device)

def normalize_loudness(waveform):
    rms = waveform.pow(2).mean().sqrt()
    target_rms = 0.1  # Empirical; tune as needed
    if rms > 0:
        waveform = waveform * (target_rms / rms)
    return waveform

def get_embedding(path):
    waveform = preprocess_audio(path)
    waveform = normalize_loudness(waveform)
    with torch.no_grad():
        emb = spkrec({"waveform": waveform, "sample_rate": 16000})
        emb = torch.tensor(emb, dtype=torch.float32, device=device)
        return model(emb.unsqueeze(0))  # Pass through TripletNet

# --- Compare two audio files ---
def compare_wavs(wav1, wav2):
    emb1 = get_embedding(wav1)
    emb2 = get_embedding(wav2)
    distance = torch.nn.functional.cosine_similarity(emb1, emb2)
    return distance.item()

# --- Entry Point ---
if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python test_tripletnet.py <wav_file_1> <wav_file_2>")
    #     sys.exit(1)

    wav1 = "D:/SPECIAL/Jarvis/voice_data/anchor/stitched_0.wav"
    wav2 = "D:/SPECIAL/Jarvis/utterance_0.wav"

    if not os.path.exists(wav1) or not os.path.exists(wav2):
        print("One or both files not found.")
        sys.exit(1)

    score = compare_wavs(wav1, wav2)
    print(f"Similarity score : {score:.4f}")
    if score > 0.8:
        print("✔️ Likely same speaker")
    else:
        print("❌ Likely different speakers")
