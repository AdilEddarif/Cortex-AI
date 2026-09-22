# CortexAI

A persistent, brain-inspired artificial cognitive system: around 20 cooperating modules
(brainstem, thalamic router, attention, global workspace, working and long-term memory, emotion,
goals, inner thought, language, speech, action selection, safety, prediction, imagination,
sleep/dreams, self-model, world model, metacognition, vision, audition) running a continuous
perception → cognition → action loop. It makes **no claim of consciousness**. See
[ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

```bash
pip install -e .[dev]            # core deps are already present on this machine
python -m pytest                 # 28 tests, offline and deterministic

python -m organism serve         # continuous organism + dashboard → http://127.0.0.1:8765
                                 #   interactive face → http://127.0.0.1:8765/face
                                 #   live brain map   → http://127.0.0.1:8765/brain
python -m organism chat --thoughts   # terminal conversation, showing inner thoughts
python -m organism experiment    # ablation experiments A–D → data/experiments/*.md
python -m organism experiment all --neural   # the same protocol with the local LLM
python -m organism status        # persisted identity, memories, event counts
```

Language uses Ollama `qwen3.5:4b` automatically when it is available, otherwise a deterministic
symbolic layer (`--llm rule`). Memory uses `nomic-embed-text` embeddings, falling back to hashing.
Everything else is configurable in `config/organism.toml` or via `ORGANISM_<SECTION>__<KEY>`
environment variables.

Optional senses: `ORGANISM_VISION__ENABLED=true` (server camera, YOLO), `ORGANISM_AUDIO__MICROPHONE=true`
(needs `pip install sounddevice`), `ORGANISM_SPEECH__TTS=kokoro`. The dashboard can also send
webcam frames, push-to-talk recordings and image uploads, and speak replies with the browser voice.

Ablate modules at runtime: `python -m organism serve --disable memory,self_model`.

## The face (`/face`)

A living particle face (~600 dots in a plexus mesh) wired to the organism:

* **Eyes = vision.** Webcam frames stream to the visual system (YOLO26n on the GPU, ~30-50 ms/frame).
  Pupils follow the person it detects, with browser-side motion tracking for smooth saccades between
  frames. A dotted "optic nerve" carries a pulse per processed frame. Percepts reach cognition only
  when the scene changes stably (hysteresis), and it may greet someone who appears.
* **Mouth = speech.** Every `SPEECH_GENERATED` event is voiced (browser voice, or Kokoro audio with
  amplitude lip-sync) and lip-synced with per-letter visemes; the caption highlights the current word.
* **Ears.** Continuous speech recognition (Chrome/Edge) feeds the auditory channel and pauses while it
  speaks; the ear dots glow with microphone level. Other browsers get push-to-talk (server Whisper).
* **Expression = internal state.** Smile follows valence, brows follow curiosity/fear/frustration, eye
  width follows urgency, pupils dilate with arousal, blinks speed up with fatigue; it closes its
  eyes asleep, turns violet with rapid eye jitter when dreaming, and ripples across the forehead on
  each thought.

Ask it to **smile, laugh, wink, frown, look surprised, close your eyes, look left/up**: facial
expressions are real actions (`express`) chosen by the organism and rendered by the face. **Go to
sleep** is also a real action: it closes its eyes, stays asleep for at least 90 s (or until spoken to),
replays memories and dreams.

## The brain map (`/brain`)

A live, anatomically-inspired map of the modules: regions light up as they emit and receive events,
and pulses travel along the real subscription wiring through the thalamic bus. Workspace broadcasts
show as ignition rings. Beside the map are a hypnogram (awake / REM-like / drowsy / NREM-like),
activity traces from measured event flow, the current workspace contents, dream content while
dreaming, and click-to-inspect for any region's live state. In sleep the sensory regions dim with
thalamic gating while the memory/sleep loop keeps working.

## Layout

```
organism/
  core/         events (schemas), bus (thalamus), module base, clock, trace, redaction
  models/       LLM router (Ollama / Anthropic / symbolic fallback), embeddings
  modules/      one file per cognitive system
  persistence/  SQLite store
  experiments/  ablation runner and metrics
  api/          FastAPI server + static dashboard
tests/          unit + integration tests (architectural guarantees)
```

## Performance note

Language-model calls take a few seconds on a consumer GPU and are only needed for
general-knowledge questions. If a call is slow or fails, it times out and falls back to the
symbolic layer, so cognition never stops. Symbolic mode replies instantly.
