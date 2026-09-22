"""Speech system: thought -> communicative intention -> language -> (safety filter) -> audio.

Only executes *approved* speak/ask actions. Speech output is also fed back as a
self-generated signal (efference copy), so the organism knows what it said without
mistaking its own voice for an external speaker.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ..core.events import ActionPayload, ActionType, Event, EventType, Modality, SpeechPayload
from ..core.module import CognitiveModule
from ..core.util import new_id, truncate


class Speech(CognitiveModule):
    name = "speech"
    subscriptions = (EventType.ACTION_APPROVED,)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.speech
        self.outputs: list[dict] = []
        self._tts = None
        self._tts_failed = False

    async def start(self) -> None:
        self.respond("executor.speak", _true)
        self.respond("executor.ask", _true)

    async def handle(self, event: Event) -> None:
        p = event.data(ActionPayload)
        if p.spec.action not in (ActionType.SPEAK, ActionType.ASK):
            return
        params = p.spec.params
        intention = params.get("intention") or ("ask_clarification" if p.spec.action == ActionType.ASK else "answer")
        if intention == "recite" and params.get("text"):  # words already fixed by a plan (e.g. counting aloud)
            gen = {"text": str(params["text"]), "grounding": ["plan"], "model": "rule"}
        else:
            gen = await self.ask_one("language.generate", {
                "intention": intention, "analysis": params.get("analysis"), "thought": params.get("thought"),
                "content": params.get("content", ""), "doing": params.get("doing"),
            }, default=None)
        if not gen:
            self._done(p, {"outcome": "failed", "detail": "no language system available"})
            return
        text = gen.get("text", "").strip()
        if not text:
            self._done(p, {"outcome": "success", "detail": "decided to stay silent"})
            return
        checked = await self.ask_one("safety.check_utterance", {"text": text}, default={"text": text, "issues": []})
        text = checked["text"]
        audio_path = await self._synthesise(text)
        addressed = params.get("addressed_to", "user")
        self.emit(EventType.SPEECH_GENERATED,
                  SpeechPayload(text=text, intention=intention, addressed_to=addressed, decision_id=p.decision_id,
                                in_reply_to=p.spec.about_event, grounding=gen.get("grounding", []),
                                knowledge_gap=bool(gen.get("knowledge_gap")), audio_path=audio_path),
                  summary=f'I said: "{truncate(text, 150)}"', modality=Modality.MOTOR, nominated=True,
                  salience=0.25, confidence=p.spec.confidence)
        self.outputs = ([{"t": self.now(), "text": text, "intention": intention, "model": gen.get("model"),
                          "grounding": gen.get("grounding", [])}] + self.outputs)[:30]
        self._done(p, {"outcome": "success", "text": text, "model": gen.get("model"),
                       "issues": checked.get("issues", [])})

    def _done(self, p: ActionPayload, result: dict) -> None:
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed", result=result),
                  summary=f"Executed {p.spec.action.value}: {result.get('outcome')}", modality=Modality.MOTOR)

    async def _synthesise(self, text: str) -> str | None:
        if self.cfg.tts != "kokoro" or self._tts_failed:
            return None
        try:
            return await asyncio.to_thread(self._kokoro, text)
        except Exception as exc:
            self._tts_failed = True
            self.log.warning("TTS unavailable (%r); continuing with text-only speech", exc)
            return None

    def _kokoro(self, text: str) -> str:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        if self._tts is None:
            self._tts = KPipeline(lang_code="a")
        chunks = [audio for _gs, _ps, audio in self._tts(text, voice=self.cfg.voice)]
        out = Path(self.settings.data_dir) / "speech" / f"{new_id('utt_')}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, np.concatenate(chunks), 24000)
        return str(out)

    def snapshot(self) -> dict:
        return {"outputs": self.outputs[:12], "tts": self.cfg.tts}


async def _true(_q: dict) -> bool:
    return True
