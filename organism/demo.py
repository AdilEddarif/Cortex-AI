"""A scripted, reproducible tour of CortexAI (``python -m organism demo``).

Eleven short scenes, each showing one claim of the architecture, with the internal events that
produced each reply printed next to it: what won the workspace, the actual arousal behind a
verbal report, the face's actions, dreams during sleep, the boot count after a restart. It runs on
the virtual clock with the deterministic language layer, so every run tells the same story.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

from .config import offline_settings
from .core.events import Event, EventType
from .organism import Organism

C = {"scene": "\033[1;36m", "you": "\033[1;37m", "ai": "\033[1;32m", "inner": "\033[2;37m", "off": "\033[0m"}


class Tour:
    def __init__(self, pace: float = 0.0, color: bool = True):
        self.pace, self.color = pace, color and sys.stdout.isatty()
        self.lines: list[tuple[str, str]] = []   # (kind, text) for the Markdown transcript
        self.events: list[Event] = []
        self.org: Organism | None = None
        self.data = Path(tempfile.mkdtemp(prefix="cortex_demo_"))

    # ------------------------------------------------------------------ output
    def out(self, kind: str, text: str) -> None:
        self.lines.append((kind, text))
        prefix = {"scene": "\n", "you": "  You: ", "ai": "  CortexAI: ", "inner": "      · "}.get(kind, "")
        s = prefix + text
        print(f"{C[kind]}{s}{C['off']}" if self.color and kind in C else s, flush=True)
        if self.pace:
            time.sleep(self.pace * (1.6 if kind in ("scene", "ai") else 0.6))

    def since(self, i: int, t: EventType) -> list[Event]:
        return [e for e in self.events[i:] if e.type == t]

    # ------------------------------------------------------------------ helpers
    async def boot(self, offline_s: float = 1.0) -> None:
        self.org = await Organism(offline_settings(self.data), offline_seconds=offline_s).start()
        self.org.add_observer(self.events.append)
        await self.org.tick(2)

    async def say(self, text: str) -> str:
        self.out("you", text)
        ans = await self.org.converse(text) or "(stays silent)"
        self.out("ai", ans)
        return ans

    def inner_thoughts(self, i: int, n: int = 1) -> None:
        for e in self.since(i, EventType.THOUGHT_GENERATED)[-n:]:
            self.out("inner", f"inner thought: {e.payload['content']}")

    # ------------------------------------------------------------------ the story
    async def run(self) -> None:
        self.out("scene", "1. Waking up: a persistent individual starts its life")
        await self.boot()
        ident = self.org.modules["self_model"].identity
        self.out("inner", f"boot #{ident.get('boot_count')}, organism id {str(ident.get('id'))[:8]}")
        self.inner_thoughts(0)

        self.out("scene", "2. Memory: what you tell it is encoded by importance")
        await self.say("Hi, I'm Ada.")
        await self.say("I just bought a telescope to watch the planets.")
        await self.say("My favourite planet is Saturn.")
        n = self.org.modules["memory"].snapshot()["counts"]
        self.out("inner", f"long-term memory: {n.get('episodic', 0)} episodes, {n.get('semantic', 0)} facts")

        self.out("scene", "3. Attention: one salient sound among many faint ones")
        i = len(self.events)
        self.org.hear_sound("a glass shattering", -3.0)
        for faint in ["a faint hum", "a quiet tick", "a soft rustle", "a distant murmur", "a light drip"]:
            self.org.hear_sound(faint, -48.0)
        await self.org.tick(1)
        ws = self.since(i, EventType.WORKSPACE_UPDATED)
        if ws:
            self.out("inner", f"6 sounds arrive at once; the workspace broadcasts: {ws[0].payload.get('focus')}")
        await self.say("What did you just hear?")

        self.out("scene", "4. Perception and prediction: it expects the world to stay put")
        self.org.see_objects(["person", "cup", "book", "lamp"])
        await self.org.tick(3)
        await self.say("What do you see?")
        self.org.see_objects(["person", "book", "lamp"])
        i = len(self.events)
        await self.org.tick(2)
        for e in self.since(i, EventType.PREDICTION_ERROR)[:1]:
            self.out("inner", f"prediction error {e.payload['error']:.2f}: {e.summary}")
        await self.say("Did anything change?")

        self.out("scene", "5. A body: it knows it is a face, and requests become real actions")
        await self.say("Do you have eyes?")
        await self.say("Can you walk around?")
        i = len(self.events)
        await self.say("Close your eyes, count to 5 out loud, then open your eyes.")
        await self.org.tick(12)
        for e in self.events[i:]:
            if e.type == EventType.ACTION_EXECUTED and e.payload["spec"]["action"] == "express":
                self.out("inner", f"face: {e.payload['result'].get('detail')}")
            elif e.type == EventType.SPEECH_GENERATED and e.payload["text"].startswith("One"):
                self.out("ai", e.payload["text"])

        self.out("scene", "6. Introspection: reports come from the actual internal state")
        ans = await self.say("How alert are you?")
        bs = self.org.modules["brainstem"].body
        self.out("inner", f"actual brainstem arousal right now: {bs.arousal:.2f}  (the report said: {ans.split('(')[-1].rstrip(').')})")

        self.out("scene", "7. Honesty: it does not invent what it was never told")
        await self.say("What is my dog's name?")
        await self.say("What is a telescope?")
        self.out("inner", "general knowledge comes from its 'books' (the language model); here it runs without one")

        self.out("scene", "8. Sleep: memories are replayed, consolidated and dreamt; dreams are tagged as not real")
        i = len(self.events)
        await self.say("Go to sleep now.")
        await self.org.tick(25)
        for e in self.since(i, EventType.DREAM_CONTENT)[:1]:
            self.out("inner", f"dream: {e.payload.get('narrative')}")
        for e in self.since(i, EventType.MEMORY_CONSOLIDATED)[:1]:
            self.out("inner", f"consolidation: {e.summary}")
        await self.org.hear("Wake up please!")
        await self.org.tick(2)
        await self.say("What did we talk about before you fell asleep?")

        self.out("scene", "9. Restart: it is switched off for a day and comes back as the same individual")
        await self.org.stop()
        await self.boot(offline_s=86400)
        await self.say("How many times have you been started?")
        await self.say("What is my name?")
        await self.say("What did we talk about yesterday?")

        self.out("scene", "10. Its own mind: thoughts it had, not scripted replies")
        await self.org.tick(40)
        await self.say("What are you thinking right now?")

        self.out("scene", "11. Plasticity: it learns what your words mean, and what is worth attending to")
        await self.say("tell me a joke")
        await self.say("dance for me")
        before = dict(self.org.modules["attention"].learned)
        await self.say("when I say hop, close your eyes")
        i = len(self.events)
        await self.say("hop")
        await self.org.tick(2)
        for e in self.since(i, EventType.ACTION_EXECUTED):
            if e.payload["spec"]["action"] == "express":
                self.out("inner", f"face: {e.payload['result'].get('detail')} (from a skill it was taught)")
        await self.say("what have you learned?")
        att = self.org.modules["attention"]
        moved = sorted(((k, att.learned[k] - before[k]) for k in before), key=lambda kv: -abs(kv[1]))[:2]
        self.out("inner", "attention weights after this conversation: "
                 + ", ".join(f"{k} {d:+.3f}" for k, d in moved)
                 + f" ({att.learning['mattered']} items mattered, {att.learning['ignored']} led nowhere)")
        await self.org.stop()

    def markdown(self) -> str:
        L = ["# CortexAI demo transcript", "",
             "Generated by `python -m organism demo` (virtual clock, deterministic language layer). "
             "Lines marked · are internal events that produced the replies, not speech.", ""]
        for kind, text in self.lines:
            if kind == "scene":
                L += ["", f"## {text}", ""]
            elif kind == "you":
                L.append(f"**You:** {text}  ")
            elif kind == "ai":
                L.append(f"**CortexAI:** {text}  ")
            elif kind == "inner":
                L.append(f"<sub>· {text}</sub>  ")
        return "\n".join(L) + "\n"


def run_demo(pace: float = 0.0, color: bool = True, transcript: Path | None = None) -> Tour:
    tour = Tour(pace=pace, color=color)
    asyncio.run(tour.run())
    if transcript is not None:
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(tour.markdown(), encoding="utf-8")
    return tour
