"""Auditory system (and the text channel).

    microphone / recording -> features (loudness, zero-crossing rate, spectral flatness,
    speech-band energy) -> {silence | environmental sound | speech} -> ASR (faster-whisper)
    -> structured percept -> language (speech) / attention (sounds)

Not all audio is language: sounds are percepts in their own right (a loud sudden sound is
salient and urgent). Typed text arrives through the same sensory interface as a ``text``
percept so that the rest of the organism treats it as perception, not as a function call.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np

from ..core.events import AudioData, Event, EventType, Modality, PerceptPayload, TextData
from ..core.module import CognitiveModule
from ..core.util import truncate

SR = 16000


def audio_features(x: np.ndarray, sr: int = SR) -> dict:
    x = x.astype(np.float32)
    if x.size == 0:
        return {"db": -120.0, "zcr": 0.0, "flatness": 1.0, "speech_band": 0.0, "duration": 0.0}
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    db = 20 * np.log10(rms)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    band = float(spec[(freqs >= 300) & (freqs <= 3400)].sum() / spec.sum())
    return {"db": round(db, 2), "zcr": round(zcr, 4), "flatness": round(flatness, 4), "speech_band": round(band, 3),
            "duration": round(x.size / sr, 2)}


def classify(f: dict, silence_db: float) -> str:
    if f["db"] < silence_db:
        return "silence"
    if f["speech_band"] > 0.55 and f["flatness"] < 0.45 and 0.02 < f["zcr"] < 0.35 and f["duration"] > 0.3:
        return "speech"
    return "sound"


class Audition(CognitiveModule):
    name = "audition"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.audio
        self._asr = None
        self._asr_failed = False
        self._mic_task: asyncio.Task | None = None
        self.recent: list[dict] = []

    async def start(self) -> None:
        if self.cfg.microphone and not self.ctx.clock.virtual:
            self._mic_task = asyncio.create_task(self._mic_loop(), name="microphone")

    async def stop(self) -> None:
        if self._mic_task:
            self._mic_task.cancel()

    def _remember(self, kind: str, text: str) -> None:
        self.recent = ([{"t": self.now(), "kind": kind, "text": text}] + self.recent)[:20]

    # ------------------------------------------------------------------ text channel
    def ingest_text(self, text: str, speaker: str = "user", channel: str = "console") -> Event:
        self._remember("text", f"{speaker}: {text}")
        return self.emit(EventType.PERCEPTION,
                         PerceptPayload(modality=Modality.TEXT, sensor=channel, description=f'{speaker} wrote: "{text}"',
                                        text=TextData(text=text, speaker=speaker, channel=channel)),
                         summary=f'{speaker} wrote: "{truncate(text, 140)}"', modality=Modality.TEXT,
                         salience=0.7, confidence=0.99)

    # ------------------------------------------------------------------ audio
    async def ingest_audio_bytes(self, data: bytes, speaker: str | None = None) -> Event | None:
        samples = await asyncio.to_thread(self._decode, data)
        return await self.ingest_samples(samples, speaker=speaker)

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        try:
            from faster_whisper.audio import decode_audio
            return decode_audio(io.BytesIO(data), sampling_rate=SR)
        except Exception:
            import soundfile as sf
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SR:
                idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
                x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
            return x

    async def ingest_samples(self, x: np.ndarray, speaker: str | None = None, sensor: str = "microphone") -> Event | None:
        f = audio_features(x)
        kind = classify(f, self.cfg.silence_db)
        transcript, lang = None, None
        if kind == "speech":
            transcript, lang = await self._transcribe(x)
            if not transcript:
                kind = "sound"
        if kind == "silence":
            self._remember("silence", f"{f['duration']}s of silence")
            return None
        label = None
        if kind == "sound":
            label = "a loud sudden sound" if f["db"] > self.cfg.loud_db else (
                "a tonal sound" if f["flatness"] < 0.2 else "background noise")
        data = AudioData(kind=kind, transcript=transcript, language=lang, speaker=speaker if kind == "speech" else None,
                         loudness_db=f["db"], duration_s=f["duration"], sound_label=label)
        desc = f'{speaker or "someone"} said: "{transcript}"' if kind == "speech" else f"I heard {label} ({f['db']:.0f} dB)"
        loud = kind == "sound" and f["db"] > self.cfg.loud_db
        self._remember(kind, desc)
        return self.emit(EventType.PERCEPTION,
                         PerceptPayload(modality=Modality.AUDIO, sensor=sensor, description=desc, audio=data),
                         summary=desc, modality=Modality.AUDIO,
                         salience=0.85 if loud else 0.6 if kind == "speech" else 0.3,
                         urgency=0.7 if loud else 0.0, confidence=0.8 if kind == "speech" else 0.6,
                         nominated=kind == "sound")

    def ingest_sound_event(self, label: str, loudness_db: float = -30.0, confidence: float = 0.7) -> Event:
        """Structured sound percept (e.g. from an external sound classifier or an experiment)."""
        data = AudioData(kind="sound", loudness_db=loudness_db, duration_s=1.0, sound_label=label)
        loud = loudness_db > self.cfg.loud_db
        self._remember("sound", label)
        return self.emit(EventType.PERCEPTION,
                         PerceptPayload(modality=Modality.AUDIO, sensor="microphone", description=f"I heard {label}",
                                        audio=data),
                         summary=f"I heard {label}", modality=Modality.AUDIO, salience=0.85 if loud else 0.45,
                         urgency=0.7 if loud else 0.1, confidence=confidence, nominated=True)

    async def _transcribe(self, x: np.ndarray) -> tuple[str | None, str | None]:
        if self._asr_failed:
            return None, None
        try:
            return await asyncio.to_thread(self._whisper, x)
        except Exception as exc:
            self._asr_failed = True
            self.log.warning("speech recognition unavailable (%r)", exc)
            return None, None

    def _whisper(self, x: np.ndarray) -> tuple[str | None, str | None]:
        if self._asr is None:
            from faster_whisper import WhisperModel
            device = self.cfg.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            self._asr = WhisperModel(self.cfg.asr_model, device=device,
                                     compute_type="int8_float16" if device == "cuda" else "int8")
        segments, info = self._asr.transcribe(x, vad_filter=True, beam_size=1)
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
        return (text or None), info.language

    async def _mic_loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            self.log.warning("microphone requested but 'sounddevice' is not installed")
            return
        chunk = int(SR * 0.5)
        buf: list[np.ndarray] = []
        silent_chunks = 0
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
            while True:
                data, _ = await asyncio.to_thread(stream.read, chunk)
                x = data[:, 0].copy()
                loud = audio_features(x)["db"] >= self.cfg.silence_db
                if loud:
                    buf.append(x)
                    silent_chunks = 0
                elif buf:
                    silent_chunks += 1
                    if silent_chunks >= 2 or len(buf) > 40:  # end of an utterance / sound event
                        await self.ingest_samples(np.concatenate(buf))
                        buf, silent_chunks = [], 0

    def snapshot(self) -> dict:
        return {"microphone": self.cfg.microphone, "asr_model": self.cfg.asr_model, "asr_failed": self._asr_failed,
                "recent": self.recent[:10]}
