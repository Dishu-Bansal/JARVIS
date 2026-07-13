import os
import torch
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from torch.nn.functional import cosine_similarity
from torch.nn import TripletMarginLoss

# Load ECAPA pre-trained model
model = ECAPA_TDNN(input_size=80, lin_neurons=192)
model.eval()

# Feature extractor
def get_embedding(file):
    signal, fs = torchaudio.load(file)
    if signal.shape[0] > 1:
        signal = signal[0:1, :]  # Convert to mono
    feats = torchaudio.compliance.kaldi.fbank(signal, num_mel_bins=80, sample_frequency=fs)
    with torch.no_grad():
        emb = model(feats.unsqueeze(0))
    return emb.squeeze()

# Load samples
anchor_files = [f'data/anchor/{f}' for f in os.listdir('data/anchor')]
positive_files = [f'data/positive/{f}' for f in os.listdir('data/positive')]
negative_files = [f'data/negative/{f}' for f in os.listdir('data/negative')]

# Simple loop for contrastive (triplet) training
loss_fn = TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    total_loss = 0
    for i in range(min(len(anchor_files), len(positive_files), len(negative_files))):
        anc = get_embedding(anchor_files[i])
        pos = get_embedding(positive_files[i])
        neg = get_embedding(negative_files[i])
        loss = loss_fn(anc.unsqueeze(0), pos.unsqueeze(0), neg.unsqueeze(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
