# JARVIS — Real-Time Conversational Voice AI Agent

A personal, local-first voice assistant that listens, decides when you're actually done talking, verifies it's really you, cleans up the audio, transcribes it, hands it to an LLM agent that can call tools, and speaks the answer back — all over a single low-latency WebSocket connection.

> **Note for recruiters / reviewers:** this repo is a personal R&D sandbox, not a polished product repo. Most of the files here are throwaway experiments — voice-cloning attempts, dataset scratch files, one-off test scripts, and an earlier abandoned raw-socket server. **The only file that matters is [`server2.py`](./server2.py).** Everything below describes what that file does and why; the rest of the repo is left visible on purpose so you can see the iteration process, but it's not meant to be read line by line.

---

## Start here: `server2.py`

This is the real-time pipeline. If you only look at one thing in this repo, look at this file.

```
Mic (client) ──▶ WebSocket ──▶ Silero VAD (endpointing) ──▶ Speaker verification
                                                                      │
Piper TTS (streamed) ◀── Gemini agent (tool calling) ◀──     Speech-to-Text
     |                                                                ▲
  Speaker (client)                                            DeepFilterNet denoising
```

### What it's doing, piece by piece

- **Real-time audio transport** — a FastAPI/`uvicorn` WebSocket endpoint (`/ws/audio`) streams raw PCM audio in both directions instead of batching requests, so the agent can react while the user is still talking.
- **Voice Activity Detection & endpointing** — Silero VAD runs on ~32ms frames with a pre-roll ring buffer (so the first syllable isn't clipped) and a silence-duration threshold to decide when an utterance has actually ended, rather than transcribing on every pause.
- **Noise filtering** — incoming audio is denoised with DeepFilterNet (resampled 16kHz ↔ 48kHz around the model) before it's verified or transcribed.
- **Speaker verification (diarization-adjacent)** — every utterance is checked against a reference voice embedding (Resemblyzer + a custom-trained triplet network, see `VoiceTraining/`) before it's allowed to reach the LLM, so the agent only responds to its enrolled user.
- **Speech-to-Text** — utterances are transcribed via a streaming STT backend (the code shows an earlier `faster-whisper` implementation still in place, later swapped for a Soniox real-time integration — left visible intentionally to show that iteration).
- **Playback-aware gating** — while the agent is speaking, incoming mic audio is ignored rather than fed back into the pipeline, avoiding the assistant hearing itself. (This is a simple gate, not full duplex barge-in — noted honestly below.)
- **LLM agent with tool calling** — a Gemini-based chat session is wired up with real tools: reading on-screen text, typing text back into the OS, a semantic "find and open this file" tool that calls out to a separate local service, and a nested web-search sub-agent. This is the "AI agent," not just an LLM call — it decides which tool to invoke based on the conversation.
- **Structured output protocol** — responses are wrapped in custom `<speak>` / `<screen>` / `<type>` tags and parsed with a small extractor, so the same LLM turn can produce a short spoken reply and a longer on-screen answer without the two getting mixed up. An early, hand-rolled version of an output-eval / parsing harness.
- **Low-latency streaming TTS** — replies are synthesized locally with Piper and streamed back over the same WebSocket in chunks as they're generated, instead of waiting for the full clip.
- **Producer/consumer concurrency** — the asyncio WebSocket loop stays responsive by handing finished utterances off to a background thread + queue for transcription and downstream processing, rather than blocking the event loop.

### Being upfront about the rough edges

Since this was a solo, local experiment rather than a shipped product, a few things I'd flag myself before anyone else does:
- API keys are placeholder strings. Obviously, Original keys are in my .env file, which is not uploaded here.
- Some tool integrations (`pyautogui`, `wmi`, `os.startfile`) are Windows-desktop-specific — this was built to run on my machine, not deployed as a service.
- There's commented-out code from earlier iterations (SpeechBrain, pyannote pipelines, a raw-socket version of the server) left in place rather than deleted, so the file also reads as a changelog of what I tried.
- No automated test suite or eval harness yet — the "evals" here are the `<speak>`/`<screen>` tag parser and manual testing, which is the honest next step if this went further.

---

## Everything else in this repo

Kept for my own reference, not meant to be reviewed:

| Path | What it is |
|---|---|
| `Memory/` | Scratch space for conversation memory experiments |
| `VoiceTraining/`, `myVoice/`, `otherVoice/`, `voice_data/` | Data and training code for the custom speaker-verification triplet network |
| `enrollment.py`, `train_voice.py` | Scripts used once to enroll my voice and train the verification model |
| `tripletnet_dishu*.pt`, `user_vec.npy`, `reference_embedding.npy` | Trained model weights / embeddings from that process |
| `server.py` | An earlier, abandoned raw-socket implementation before I moved to FastAPI WebSockets in `server2.py` |
| `client.py` | A basic test client used during development |
| `test.py`, `test2.py`, `*.wav`, `output.srt` | One-off debugging scripts and sample audio used while tuning VAD/denoising thresholds |

---

## Tech touched in `server2.py`

Python · FastAPI · WebSockets · asyncio · Silero VAD · DeepFilterNet · Resemblyzer · PyTorch · torchaudio · Faster-Whisper / Soniox (STT) · Google Gemini (LLM + function calling) · Piper (local streaming TTS) · threading/Queue producer-consumer pattern · real-time low-latency audio streaming
