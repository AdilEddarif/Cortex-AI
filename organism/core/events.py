"""Structured event schemas.

Every message between modules is an ``Event`` carrying provenance, timing and epistemic
metadata (confidence, salience, uncertainty...). The ``payload`` of each event type is
validated against a registered pydantic model, so modules never exchange arbitrary strings.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from .util import new_id


class EventType(StrEnum):
    # rhythm / body
    TICK = "TICK"
    BODY_STATE = "BODY_STATE"
    MODE_CHANGED = "MODE_CHANGED"
    # perception & language
    PERCEPTION = "PERCEPTION"
    UTTERANCE_UNDERSTOOD = "UTTERANCE_UNDERSTOOD"
    # attention & workspace
    ATTENTION_CHANGED = "ATTENTION_CHANGED"
    WORKSPACE_UPDATED = "WORKSPACE_UPDATED"
    WM_UPDATED = "WM_UPDATED"
    # memory
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"
    MEMORY_STORED = "MEMORY_STORED"
    MEMORY_REPLAYED = "MEMORY_REPLAYED"
    MEMORY_CONSOLIDATED = "MEMORY_CONSOLIDATED"
    # cognition
    THOUGHT_GENERATED = "THOUGHT_GENERATED"
    GOAL_CREATED = "GOAL_CREATED"
    GOAL_UPDATED = "GOAL_UPDATED"
    EMOTION_CHANGED = "EMOTION_CHANGED"
    PREDICTION_MADE = "PREDICTION_MADE"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    SIMULATION_RESULT = "SIMULATION_RESULT"
    # action
    ACTION_PROPOSED = "ACTION_PROPOSED"
    ACTION_SELECTED = "ACTION_SELECTED"
    ACTION_APPROVED = "ACTION_APPROVED"
    ACTION_REJECTED = "ACTION_REJECTED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    SPEECH_GENERATED = "SPEECH_GENERATED"
    VERBAL_REPORT = "VERBAL_REPORT"
    # sleep
    SLEEP_STARTED = "SLEEP_STARTED"
    SLEEP_ENDED = "SLEEP_ENDED"
    DREAM_STARTED = "DREAM_STARTED"
    DREAM_ENDED = "DREAM_ENDED"
    DREAM_CONTENT = "DREAM_CONTENT"
    # models of self and world
    SELF_STATE_CHANGED = "SELF_STATE_CHANGED"
    WORLD_UPDATED = "WORLD_UPDATED"
    WORLD_CONFLICT = "WORLD_CONFLICT"
    METACOGNITIVE_REPORT = "METACOGNITIVE_REPORT"
    # system
    SYSTEM_BOOT = "SYSTEM_BOOT"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    MODULE_ERROR = "MODULE_ERROR"
    COMMAND = "COMMAND"


class Modality(StrEnum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    INTERNAL = "internal"
    INTEROCEPTIVE = "interoceptive"
    MEMORY = "memory"
    EMOTION = "emotion"
    MOTOR = "motor"
    IMAGINATION = "imagination"


class Mode(StrEnum):
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    ASLEEP = "ASLEEP"
    DREAMING = "DREAMING"


class ActionType(StrEnum):
    SPEAK = "speak"
    ASK = "ask"
    WAIT = "wait"            # includes deliberate silence
    LOOK = "look"
    REMEMBER = "remember"
    INVESTIGATE = "investigate"
    SIMULATE = "simulate"
    SLEEP = "sleep"
    WAKE = "wake"
    SET_GOAL = "set_goal"
    NOTE = "note"            # sandboxed digital action (permission-controlled)
    MOVE = "move"            # no body yet: always rejected as unavailable
    EXPRESS = "express"      # facial expression (the face is the organism's visible body)
    REVOKE = "revoke"        # give up one of its own permissions (it can never grant itself one)


VOCAL_ACTIONS = frozenset({ActionType.SPEAK, ActionType.ASK})
EXTERNAL_ACTIONS = frozenset({ActionType.SPEAK, ActionType.ASK, ActionType.LOOK, ActionType.NOTE, ActionType.MOVE,
                              ActionType.EXPRESS})


# --------------------------------------------------------------------------- payloads
class TickPayload(BaseModel):
    cycle: int
    interval: float
    body: "BodyPayload"


class BodyPayload(BaseModel):
    arousal: float = 0.45
    energy: float = 1.0
    fatigue: float = 0.0
    sleep_pressure: float = 0.1
    stress: float = 0.0
    curiosity: float = 0.3
    urgency: float = 0.0
    safety: float = 1.0
    novelty: float = 0.0
    mode: Mode = Mode.AWAKE


class ModePayload(BaseModel):
    previous: Mode
    current: Mode
    reason: str


class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # normalised x1, y1, x2, y2


class SpatialRelation(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = 0.7


class VisionData(BaseModel):
    objects: list[DetectedObject] = Field(default_factory=list)
    people: int = 0
    scene: str | None = None
    spatial_relations: list[SpatialRelation] = Field(default_factory=list)
    change_score: float = 0.0
    frame_size: tuple[int, int] | None = None


class AudioData(BaseModel):
    kind: Literal["speech", "sound", "silence"]
    transcript: str | None = None
    language: str | None = None
    speaker: str | None = None
    loudness_db: float = -90.0
    duration_s: float = 0.0
    sound_label: str | None = None


class TextData(BaseModel):
    text: str
    speaker: str = "user"
    channel: str = "console"


class PerceptPayload(BaseModel):
    modality: Modality
    sensor: str
    description: str
    vision: VisionData | None = None
    audio: AudioData | None = None
    text: TextData | None = None
    self_generated: bool = False


class Fact(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = 0.7


class LanguageAnalysis(BaseModel):
    text: str
    speaker: str = "user"
    intent: Literal["question", "command", "statement", "greeting", "farewell", "other"] = "statement"
    is_question: bool = False
    addressed_to_self: bool = True
    asks_about: str | None = None
    topic: str = ""
    entities: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    sentiment: float = 0.0
    command: str | None = None
    command_arg: str | None = None
    # A request made of several steps ("close your eyes, count to 10, then open them"), in order:
    # [{"command": ..., "arg": ..., "text": the words of that step}]. ``command`` is then "sequence".
    steps: list[dict] = Field(default_factory=list)
    hostile: bool = False   # a threat or hostility aimed at the organism ("I will kill you")
    apology: bool = False   # an apology or retraction ("sorry, I was just testing you")
    affirm: str | None = None   # "yes" or "no": a bare answer, with nothing else in it
    insult: bool = False    # an insult aimed at the organism ("you are useless")
    request: str = ""       # something asked for that maps to no capability ("dance", "book a flight")
    unsupported: bool = False


class UtterancePayload(BaseModel):
    analysis: LanguageAnalysis
    percept_id: str
    modality: Modality = Modality.TEXT


class AttentionItem(BaseModel):
    event_id: str
    event_type: EventType
    source: str
    summary: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    event: dict[str, Any] = Field(default_factory=dict)


class AttentionPayload(BaseModel):
    selected: list[AttentionItem] = Field(default_factory=list)
    threshold: float = 0.0
    candidates: int = 0
    mode: Mode = Mode.AWAKE


class WorkspaceItem(BaseModel):
    event_id: str
    event_type: EventType
    source: str
    modality: Modality
    summary: str
    score: float
    entered_cycle: int
    components: dict[str, float] = Field(default_factory=dict)
    event: dict[str, Any] = Field(default_factory=dict)


class WorkspacePayload(BaseModel):
    cycle: int
    items: list[WorkspaceItem] = Field(default_factory=list)
    new_item_ids: list[str] = Field(default_factory=list)
    focus: str | None = None
    integrated: bool = True   # False when the global workspace is ablated (local relay)


class WMPayload(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    evicted: list[str] = Field(default_factory=list)


class MemoryRef(BaseModel):
    id: str
    kind: str
    content: str
    score: float = 0.0
    timestamp: float = 0.0
    importance: float = 0.0
    confidence: float = 1.0
    source: str = ""
    episode_id: str | None = None

    @property
    def is_dream(self) -> bool:
        return self.kind == "dream"


class MemoryRetrievedPayload(BaseModel):
    cue: str
    memories: list[MemoryRef]


class MemoryStoredPayload(BaseModel):
    memory: MemoryRef


class MemoryReplayedPayload(BaseModel):
    memories: list[MemoryRef]


class ConsolidationPayload(BaseModel):
    replayed: int = 0
    abstractions: list[str] = Field(default_factory=list)
    strengthened: int = 0
    pruned: int = 0
    dreams: int = 0
    duration_s: float = 0.0


class ActionSpec(BaseModel):
    action: ActionType
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.7
    expected_outcome: str = ""
    goal_id: str | None = None
    about_event: str | None = None
    value: float = 0.0
    simulated: bool = False
    context_modalities: list[str] = Field(default_factory=list)


class ActionPayload(BaseModel):
    decision_id: str
    spec: ActionSpec
    status: Literal["proposed", "selected", "approved", "rejected", "executed", "failed"]
    reason: str = ""
    result: dict[str, Any] = Field(default_factory=dict)


class ThoughtPayload(BaseModel):
    content: str
    kind: Literal[
        "observation", "inference", "prediction", "question", "decision", "reflection",
        "mind_wandering", "plan",
    ] = "inference"
    confidence: float = 0.6
    chain_id: str = ""
    step: int = 0
    trigger: str = ""
    grounded_in: list[str] = Field(default_factory=list)
    proposed_action: ActionSpec | None = None
    stop_reason: str | None = None


class Goal(BaseModel):
    id: str = Field(default_factory=lambda: new_id("g_"))
    description: str
    priority: float = 0.5
    origin: Literal["survival", "intrinsic", "social", "task", "exploration", "user"] = "task"
    kind: Literal["primary", "secondary", "temporary", "intrinsic", "maintenance"] = "temporary"
    status: Literal["active", "achieved", "failed", "abandoned", "expired"] = "active"
    created_at: float = 0.0
    deadline: float | None = None
    expected_reward: float = 0.5
    required_actions: list[str] = Field(default_factory=list)
    progress: float = 0.0
    related_event: str | None = None


class GoalPayload(BaseModel):
    goal: Goal
    change: str = "created"


class EmotionPayload(BaseModel):
    state: dict[str, float]
    valence: float
    intensity: float
    deltas: dict[str, float] = Field(default_factory=dict)
    cause: str = ""


class SpeechPayload(BaseModel):
    text: str
    intention: str
    addressed_to: str = "user"
    decision_id: str = ""
    in_reply_to: str | None = None
    grounding: list[str] = Field(default_factory=list)
    knowledge_gap: bool = False
    audio_path: str | None = None


class VerbalReportPayload(BaseModel):
    """A verbal claim about an internal state, stored next to the state it describes.
    The report never writes back into the state."""
    topic: str
    report: str
    state_snapshot: dict[str, Any]


class PredictionPayload(BaseModel):
    prediction_id: str
    target: str
    predicted: Any
    confidence: float = 0.5
    expires_at: float | None = None


class PredictionErrorPayload(BaseModel):
    prediction_id: str
    target: str
    predicted: Any
    observed: Any
    error: float


class SimulationPayload(BaseModel):
    scenario: str
    action: ActionSpec | None = None
    predicted_outcome: str
    success_probability: float = 0.5
    value: float = 0.0
    risks: list[str] = Field(default_factory=list)
    narrative: str = ""
    imagined: bool = True


class DreamPayload(BaseModel):
    dream_id: str
    narrative: str
    fragments: list[str] = Field(default_factory=list)
    step: int = 0


class SelfStatePayload(BaseModel):
    changes: dict[str, Any]
    reason: str = ""


class WorldPayload(BaseModel):
    entity_id: str
    changes: dict[str, Any]


class WorldConflictPayload(BaseModel):
    entity_id: str
    attribute: str
    believed: Any
    believed_confidence: float
    observed: Any
    observed_confidence: float


class MetaPayload(BaseModel):
    kind: Literal[
        "uncertainty", "knowledge_gap", "conflict", "focus", "source_monitoring",
        "performance", "loop_detected", "module_failure",
    ]
    content: str
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)


class CommandPayload(BaseModel):
    command: str
    args: dict[str, Any] = Field(default_factory=dict)


class SystemPayload(BaseModel):
    info: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    module: str
    error: str
    event_id: str | None = None


TickPayload.model_rebuild()

PAYLOADS: dict[EventType, type[BaseModel]] = {
    EventType.TICK: TickPayload,
    EventType.BODY_STATE: BodyPayload,
    EventType.MODE_CHANGED: ModePayload,
    EventType.PERCEPTION: PerceptPayload,
    EventType.UTTERANCE_UNDERSTOOD: UtterancePayload,
    EventType.ATTENTION_CHANGED: AttentionPayload,
    EventType.WORKSPACE_UPDATED: WorkspacePayload,
    EventType.WM_UPDATED: WMPayload,
    EventType.MEMORY_RETRIEVED: MemoryRetrievedPayload,
    EventType.MEMORY_STORED: MemoryStoredPayload,
    EventType.MEMORY_REPLAYED: MemoryReplayedPayload,
    EventType.MEMORY_CONSOLIDATED: ConsolidationPayload,
    EventType.THOUGHT_GENERATED: ThoughtPayload,
    EventType.GOAL_CREATED: GoalPayload,
    EventType.GOAL_UPDATED: GoalPayload,
    EventType.EMOTION_CHANGED: EmotionPayload,
    EventType.PREDICTION_MADE: PredictionPayload,
    EventType.PREDICTION_ERROR: PredictionErrorPayload,
    EventType.SIMULATION_RESULT: SimulationPayload,
    EventType.ACTION_PROPOSED: ActionPayload,
    EventType.ACTION_SELECTED: ActionPayload,
    EventType.ACTION_APPROVED: ActionPayload,
    EventType.ACTION_REJECTED: ActionPayload,
    EventType.ACTION_EXECUTED: ActionPayload,
    EventType.SPEECH_GENERATED: SpeechPayload,
    EventType.VERBAL_REPORT: VerbalReportPayload,
    EventType.SLEEP_STARTED: ModePayload,
    EventType.SLEEP_ENDED: ModePayload,
    EventType.DREAM_STARTED: ModePayload,
    EventType.DREAM_ENDED: ModePayload,
    EventType.DREAM_CONTENT: DreamPayload,
    EventType.SELF_STATE_CHANGED: SelfStatePayload,
    EventType.WORLD_UPDATED: WorldPayload,
    EventType.WORLD_CONFLICT: WorldConflictPayload,
    EventType.METACOGNITIVE_REPORT: MetaPayload,
    EventType.SYSTEM_BOOT: SystemPayload,
    EventType.SYSTEM_SHUTDOWN: SystemPayload,
    EventType.MODULE_ERROR: ErrorPayload,
    EventType.COMMAND: CommandPayload,
}

M = TypeVar("M", bound=BaseModel)


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev_"))
    type: EventType
    source: str
    timestamp: float = 0.0
    cycle: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    modality: Modality = Modality.INTERNAL
    confidence: float = 1.0
    salience: float = 0.0
    importance: float = 0.0
    urgency: float = 0.0
    emotional_value: float = 0.0     # valence in [-1, 1]
    uncertainty: float | None = None
    gain: float = 1.0                # thalamic gating applied to salience (sensory events)
    caused_by: list[str] = Field(default_factory=list)
    nominated: bool = False          # candidate for attention / the global workspace

    @model_validator(mode="after")
    def _defaults(self) -> "Event":
        if self.uncertainty is None:
            self.uncertainty = round(1.0 - self.confidence, 4)
        return self

    def data(self, model: type[M]) -> M:
        return model.model_validate(self.payload)

    @property
    def raw_salience(self) -> float:
        return self.salience / self.gain if self.gain > 0 else self.salience


def make_event(
    etype: EventType, source: str, payload: BaseModel | dict | None = None, **meta: Any
) -> Event:
    expected = PAYLOADS.get(etype)
    if isinstance(payload, BaseModel):
        if expected is not None and not isinstance(payload, expected):
            raise TypeError(f"{etype} expects payload {expected.__name__}, got {type(payload).__name__}")
        data = payload.model_dump(mode="json")
    else:
        data = dict(payload or {})
        if expected is not None:
            data = expected.model_validate(data).model_dump(mode="json")
    return Event(type=etype, source=source, payload=data, **meta)
