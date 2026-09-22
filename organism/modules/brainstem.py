"""Brainstem / arousal system.

Owns the organism's homeostatic variables and its cognitive rhythm. Each TICK starts one
cognitive cycle (attention competition -> workspace broadcast -> downstream processing).
The tick period shrinks as arousal rises, so an alert organism literally thinks faster.

Sleep follows a two-process-like model: sleep pressure accumulates while awake and
discharges during sleep; NREM-like (ASLEEP) and REM-like (DREAMING) phases alternate.
Sensory gating: in sleep the thalamic gain on external input is lowered, so only strong
or urgent stimuli wake the organism.
"""
from __future__ import annotations

import asyncio

from ..core.events import (
    ActionPayload, ActionType, BodyPayload, CommandPayload, EmotionPayload, Event, EventType,
    Modality, Mode, ModePayload, TickPayload,
)
from ..core.module import CognitiveModule
from ..core.util import clamp

SENSORY_GAIN = {Mode.AWAKE: 1.0, Mode.DROWSY: 0.7, Mode.ASLEEP: 0.3, Mode.DREAMING: 0.15}
MODE_EVENT = {
    (Mode.DROWSY, Mode.ASLEEP): EventType.SLEEP_STARTED,
    (Mode.AWAKE, Mode.ASLEEP): EventType.SLEEP_STARTED,
    (Mode.ASLEEP, Mode.DREAMING): EventType.DREAM_STARTED,
    (Mode.DREAMING, Mode.ASLEEP): EventType.DREAM_ENDED,
}


class Brainstem(CognitiveModule):
    name = "brainstem"
    subscriptions = (
        EventType.PERCEPTION, EventType.UTTERANCE_UNDERSTOOD, EventType.PREDICTION_ERROR,
        EventType.EMOTION_CHANGED, EventType.COMMAND, EventType.ACTION_APPROVED,
        EventType.THOUGHT_GENERATED, EventType.SPEECH_GENERATED,
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cfg = self.settings.brainstem
        self.body = BodyPayload(arousal=self.cfg.baseline_arousal)
        self.cycle = 0
        self._stim = 0.0
        self._load = 0
        self._mode_ticks = 0
        self._last_tick: float | None = None
        self._last_stimulus = self.now()
        self._last_social = 0.0
        self._last_interoceptive = 0.0
        self._urgent = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._sleep_started: float | None = None
        self._min_sleep_until = 0.0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        saved = self.kv_load("body")
        if saved:
            self.body = BodyPayload.model_validate(saved)
            # After a restart the organism boots awake, whatever mode it was persisted in.
            self.body.mode = Mode.AWAKE
            self.cycle = self.kv_load("cycle", 0)
        self.bus.cycle = self.cycle
        self._apply_gain()
        self.respond("body.state", self._body_state)
        self.respond("executor.sleep", _true)
        self.respond("executor.wake", _true)
        if not self.ctx.clock.virtual:
            self._task = asyncio.create_task(self._rhythm(), name="brainstem-rhythm")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self.persist()

    def persist(self) -> None:
        self.kv_save("body", self.body.model_dump(mode="json"))
        self.kv_save("cycle", self.cycle)

    async def _body_state(self, _q: dict) -> dict:
        return self.body.model_dump(mode="json")

    # ------------------------------------------------------------------ rhythm
    def interval(self) -> float:
        if self.body.mode in (Mode.ASLEEP, Mode.DREAMING):
            return self.cfg.sleep_tick_s
        a = self.body.arousal
        return self.cfg.tick_max_s - (self.cfg.tick_max_s - self.cfg.tick_min_s) * a

    async def _rhythm(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._urgent.wait(), self.interval())
            except asyncio.TimeoutError:
                pass
            self._urgent.clear()
            try:
                self.tick()
            except Exception:
                self.log.exception("tick failed")

    def request_phasic_tick(self) -> None:
        self._urgent.set()

    # ------------------------------------------------------------------ inputs
    async def handle(self, event: Event) -> None:
        t = event.type
        now = self.now()
        b = self.body
        if t == EventType.PERCEPTION:
            if event.payload.get("self_generated"):
                return
            self._stim = max(self._stim, event.salience)
            self._last_stimulus = now
            self.ctx.temporal.last_external_input = now
            if b.mode in (Mode.ASLEEP, Mode.DREAMING) and (
                    event.raw_salience >= self.cfg.wake_raw_salience or event.urgency >= self.cfg.wake_urgency):
                self.wake(f"woken by a strong {event.modality.value} stimulus: {event.summary[:80]}")
            elif b.mode == Mode.DROWSY and event.salience >= 0.4:
                self.set_mode(Mode.AWAKE, "roused by a stimulus")
        elif t == EventType.UTTERANCE_UNDERSTOOD:
            self._stim = max(self._stim, event.salience)
            self._last_stimulus = self._last_social = now
            b.urgency = max(b.urgency, event.urgency)
            if b.mode in (Mode.ASLEEP, Mode.DREAMING) and event.urgency >= self.cfg.wake_urgency:
                self.wake("someone spoke to me")
            elif b.mode == Mode.DROWSY:
                self.set_mode(Mode.AWAKE, "someone spoke to me")
            if event.urgency >= 0.5:
                self.request_phasic_tick()
        elif t == EventType.PREDICTION_ERROR:
            err = float(event.payload.get("error", 0.0))
            b.novelty = max(b.novelty, err)
            self._stim = max(self._stim, 0.5 * err)
        elif t == EventType.EMOTION_CHANGED:
            s = event.data(EmotionPayload).state
            b.stress = clamp(0.7 * b.stress + 0.3 * (s.get("fear", 0) + 0.5 * s.get("frustration", 0)))
            b.curiosity = s.get("curiosity", b.curiosity)
            b.urgency = max(b.urgency, s.get("urgency", 0.0))
            b.safety = clamp(1.0 - s.get("fear", 0.0))
        elif t == EventType.COMMAND:
            cmd = event.data(CommandPayload).command
            if cmd == "sleep":
                self.fall_asleep("instructed to sleep")
            elif cmd == "wake":
                self.wake("instructed to wake")
        elif t == EventType.ACTION_APPROVED:
            spec = event.data(ActionPayload).spec
            if spec.action == ActionType.SLEEP:
                self.fall_asleep(spec.reason or "I decided to sleep")
                self._executed(event, "falling asleep")
            elif spec.action == ActionType.WAKE:
                self.wake(spec.reason or "I decided to wake")
                self._executed(event, "awake")
        else:
            self._load += 1

    def _executed(self, event: Event, result: str) -> None:
        p = event.data(ActionPayload)
        self.emit(EventType.ACTION_EXECUTED,
                  ActionPayload(decision_id=p.decision_id, spec=p.spec, status="executed",
                                result={"outcome": "success", "detail": result}),
                  summary=f"Executed {p.spec.action.value}: {result}", modality=Modality.MOTOR)

    # ------------------------------------------------------------------ the cycle
    def tick(self) -> Event:
        now = self.now()
        dt = 0.0 if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now
        cfg, b = self.cfg, self.body
        self.cycle += 1
        self.bus.cycle = self.cycle
        self._mode_ticks += 1

        awake = b.mode in (Mode.AWAKE, Mode.DROWSY)
        if awake:
            b.sleep_pressure = clamp(b.sleep_pressure + dt / cfg.wake_period_s)
            b.energy = clamp(b.energy - dt / (cfg.wake_period_s * 1.5) - 0.002 * self._load)
        else:
            b.sleep_pressure = clamp(b.sleep_pressure - dt / cfg.sleep_period_s)
            b.energy = clamp(b.energy + dt / cfg.sleep_period_s)
        b.fatigue = clamp(0.6 * b.sleep_pressure + 0.4 * (1.0 - b.energy))
        b.novelty *= 0.7
        b.urgency *= 0.6

        stim = self._stim
        if b.mode == Mode.AWAKE:
            target = (cfg.baseline_arousal + 0.3 * stim + 0.15 * b.urgency + 0.15 * b.stress
                      + 0.1 * (b.curiosity - 0.35) + 0.1 * b.novelty - 0.45 * b.fatigue)
        elif b.mode == Mode.DROWSY:
            target = 0.18 + 0.4 * stim - 0.2 * b.fatigue
        elif b.mode == Mode.ASLEEP:
            target = 0.05
        else:
            target = 0.15  # REM-like internal activation
        target = clamp(target)
        rate = 0.6 if target > b.arousal else 0.25   # phasic rise, slow tonic decay
        b.arousal = round(clamp(b.arousal + (target - b.arousal) * rate), 4)
        self._stim = 0.0
        self._load = 0

        self._transitions(now)
        self._update_temporal(now, dt)

        ev = self.emit(
            EventType.TICK, TickPayload(cycle=self.cycle, interval=self.interval(), body=b),
            summary=f"cycle {self.cycle} [{b.mode.value}] arousal={b.arousal:.2f}",
            modality=Modality.INTEROCEPTIVE,
        )
        self._interoception(now)
        if self.cycle % 20 == 0:
            self.persist()
        return ev

    def _transitions(self, now: float) -> None:
        cfg, b = self.cfg, self.body
        quiet_for = now - max(self._last_stimulus, self._last_social)
        if b.mode == Mode.AWAKE and cfg.autosleep:
            sleepy = b.arousal < cfg.drowsy_arousal and b.fatigue > 0.55
            exhausted = b.fatigue > 0.9
            if (sleepy and quiet_for > cfg.quiet_before_sleep_s) or (exhausted and quiet_for > 10):
                self.set_mode(Mode.DROWSY, "fatigue is high and nothing is happening")
        elif b.mode == Mode.DROWSY:
            if self._mode_ticks >= 3 and quiet_for > 5:
                self.set_mode(Mode.ASLEEP, "drifting off to sleep")
        elif b.mode == Mode.ASLEEP:
            if b.sleep_pressure <= 0.05 and b.energy >= 0.9 and self._mode_ticks >= 2 and now >= self._min_sleep_until:
                self.wake("rested: sleep pressure discharged")
            elif self._mode_ticks >= cfg.nrem_ticks:
                self.set_mode(Mode.DREAMING, "entering a dream-like (REM-like) phase")
        elif b.mode == Mode.DREAMING:
            if self._mode_ticks >= cfg.rem_ticks:
                self.ctx.temporal.sleep_cycles += 1
                self.set_mode(Mode.ASLEEP, "dream-like phase ended")

    def _update_temporal(self, now: float, dt: float) -> None:
        tmp = self.ctx.temporal
        tmp.previous_time = tmp.current_time or now
        tmp.current_time = now
        tmp.elapsed_since_boot = now - tmp.boot_time if tmp.boot_time else 0.0
        last_in = tmp.last_external_input
        if last_in is not None and now - last_in > self.cfg.episode_gap_s and now - tmp.episode_started > self.cfg.episode_gap_s:
            tmp.new_episode(now)

    def _interoception(self, now: float) -> None:
        """Strong bodily signals are nominated for awareness ("I feel tired")."""
        b = self.body
        if b.mode != Mode.AWAKE or now - self._last_interoceptive < 60:
            return
        if b.fatigue > 0.75:
            text, sal = f"I feel tired (fatigue {b.fatigue:.2f}, sleep pressure {b.sleep_pressure:.2f})", b.fatigue * 0.6
        elif b.energy < 0.2:
            text, sal = f"My energy is low ({b.energy:.2f})", 0.5
        else:
            return
        self._last_interoceptive = now
        self.emit(EventType.BODY_STATE, b, summary=text, modality=Modality.INTEROCEPTIVE,
                  salience=sal, urgency=0.2, nominated=True, confidence=0.95)

    # ------------------------------------------------------------------ modes
    def set_mode(self, new: Mode, reason: str) -> None:
        old = self.body.mode
        if new == old:
            return
        self.body.mode = new
        self._mode_ticks = 0
        self._apply_gain()
        payload = ModePayload(previous=old, current=new, reason=reason)
        self.emit(EventType.MODE_CHANGED, payload, summary=f"{old.value} -> {new.value}: {reason}",
                  modality=Modality.INTEROCEPTIVE)
        specific = MODE_EVENT.get((old, new))
        if specific is None and new == Mode.AWAKE and old in (Mode.ASLEEP, Mode.DREAMING):
            specific = EventType.SLEEP_ENDED
        if specific is not None:
            nominated = specific in (EventType.SLEEP_ENDED, EventType.SLEEP_STARTED)
            self.emit(specific, payload, summary=reason, modality=Modality.INTEROCEPTIVE,
                      nominated=nominated, salience=0.5 if nominated else 0.0)
        if new == Mode.ASLEEP and old in (Mode.AWAKE, Mode.DROWSY):
            self._sleep_started = self.now()
        if new == Mode.AWAKE and old in (Mode.ASLEEP, Mode.DREAMING):
            self.ctx.temporal.new_episode(self.now())
            self._sleep_started = None

    def fall_asleep(self, reason: str) -> None:
        """Deliberate or requested sleep: lasts at least ``requested_sleep_s`` unless something wakes me."""
        if self.body.mode in (Mode.AWAKE, Mode.DROWSY):
            self._min_sleep_until = self.now() + self.cfg.requested_sleep_s
            self.set_mode(Mode.ASLEEP, reason)

    def wake(self, reason: str) -> None:
        if self.body.mode != Mode.AWAKE:
            if self.body.mode == Mode.DREAMING:
                self.set_mode(Mode.ASLEEP, "dream interrupted")
            self.set_mode(Mode.AWAKE, reason)
            self.body.arousal = max(self.body.arousal, 0.6)

    def _apply_gain(self) -> None:
        self.bus.sensory_gain["*"] = SENSORY_GAIN[self.body.mode]

    # ------------------------------------------------------------------ observability
    def snapshot(self) -> dict:
        return {**self.body.model_dump(mode="json"), "cycle": self.cycle, "interval": round(self.interval(), 3),
                "gain": SENSORY_GAIN[self.body.mode], "sleep_cycles": self.ctx.temporal.sleep_cycles}

    def trace_state(self) -> dict:
        b = self.body
        return {"mode": b.mode.value, "arousal": round(b.arousal, 2), "fatigue": round(b.fatigue, 2)}


async def _true(_q: dict) -> bool:
    return True
