"""Organism configuration.

Precedence (highest first): explicit overrides passed to ``load_settings`` > environment
variables (``ORGANISM_<SECTION>__<KEY>``) > ``config/organism.toml`` > defaults below.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

MODULE_NAMES = (
    "brainstem", "attention", "workspace", "working_memory", "memory", "emotion", "goals",
    "thought", "language", "speech", "action", "safety", "prediction", "imagination",
    "sleep", "self_model", "world_model", "metacognition", "vision", "audition", "expression",
    "knowledge",
)
# Modules the organism cannot run without (the rhythm generator and the sensory text channel).
ESSENTIAL_MODULES = ("brainstem",)


class IdentityCfg(BaseModel):
    name: str = "CortexAI"
    version: str = "v1"
    description: str = (
        "an experimental brain-inspired artificial cognitive system running as software on a computer"
    )


class ClockCfg(BaseModel):
    # real: organism time follows wall time (scaled). virtual: time advances only when ticked
    # (used for tests and reproducible experiments).
    mode: Literal["real", "virtual"] = "real"
    time_scale: float = 1.0
    virtual_tick_seconds: float = 1.0


class LLMCfg(BaseModel):
    provider: Literal["auto", "ollama", "anthropic", "rule"] = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_vision_model: str = "qwen3.5:4b"
    anthropic_model: str = "claude-sonnet-5"
    timeout_s: float = 90.0
    temperature: float = 0.6
    num_ctx: int = 4096
    keep_alive: str = "30m"


class EmbeddingCfg(BaseModel):
    provider: Literal["auto", "ollama", "hash"] = "auto"
    ollama_model: str = "nomic-embed-text"
    hash_dims: int = 384


class BrainstemCfg(BaseModel):
    tick_min_s: float = 0.3          # cognitive cycle period at maximal arousal
    tick_max_s: float = 2.5          # ... at minimal arousal
    sleep_tick_s: float = 2.0
    baseline_arousal: float = 0.45
    wake_period_s: float = 1800.0    # organism-seconds awake until sleep pressure saturates
    sleep_period_s: float = 300.0    # organism-seconds of sleep to fully discharge it
    autosleep: bool = True
    drowsy_arousal: float = 0.22
    quiet_before_sleep_s: float = 45.0
    wake_urgency: float = 0.6        # stimulus urgency that wakes the organism
    wake_raw_salience: float = 0.85
    nrem_ticks: int = 6
    rem_ticks: int = 4
    episode_gap_s: float = 600.0
    requested_sleep_s: float = 90.0  # a sleep that was asked for (or decided) lasts at least this long


class AttentionCfg(BaseModel):
    # Noisy-OR weights: each component can independently make an item salient.
    weights: dict[str, float] = Field(default_factory=lambda: {
        "salience": 0.8, "novelty": 0.35, "urgency": 0.8, "goal_relevance": 0.6,
        "emotional_relevance": 0.5, "uncertainty": 0.1, "prediction_error": 0.7,
    })
    ignition_threshold: float = 0.3
    inhibition_of_return_s: float = 20.0
    candidate_ttl_cycles: int = 4


class WorkspaceCfg(BaseModel):
    capacity: int = 4
    decay: float = 0.6
    floor: float = 0.12


class WorkingMemoryCfg(BaseModel):
    capacity: int = 7
    half_life_s: float = 90.0
    dialog_turns: int = 12
    dialog_ttl_s: float = 300.0


class MemoryCfg(BaseModel):
    encode_threshold: float = 0.25
    retrieval_k: int = 5
    auto_retrieve_threshold: float = 0.35
    recency_half_life_s: float = 86400.0
    refractory_s: float = 120.0


class ThoughtCfg(BaseModel):
    max_steps: int = 3
    min_interval_s: float = 4.0
    max_chains_per_min: int = 6
    spontaneous: bool = True
    idle_ticks_before_wandering: int = 25
    deliberate_before_speaking: bool = True


class ImaginationCfg(BaseModel):
    max_candidates: int = 3


class SleepCfg(BaseModel):
    replay_batch: int = 6
    cluster_similarity: float = 0.45
    prune_importance: float = 0.08
    prune_min_age_s: float = 86400.0
    dream_fragments: int = 3


class VisionCfg(BaseModel):
    enabled: bool = False            # server-side camera loop (privacy: off by default)
    camera_index: int = 0
    fps: float = 1.0
    detector: Literal["yolo", "none"] = "yolo"
    yolo_model: str = "yolo26n.pt"
    yolo_fallback_model: str = "yolo11n.pt"
    min_confidence: float = 0.35
    change_threshold: float = 0.04
    describe_with_vlm: bool = False
    vlm_interval_s: float = 20.0
    keep_frames: bool = False


class AudioCfg(BaseModel):
    microphone: bool = False
    asr_model: str = "small"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    silence_db: float = -50.0
    loud_db: float = -12.0


class SpeechCfg(BaseModel):
    tts: Literal["none", "kokoro"] = "none"
    voice: str = "af_heart"
    max_per_minute: int = 12


class SafetyCfg(BaseModel):
    allowed_actions: list[str] = Field(default_factory=lambda: [
        "speak", "ask", "wait", "look", "remember", "investigate", "simulate",
        "sleep", "wake", "set_goal", "note", "move", "express",
    ])
    # Actions that touch the world outside the organism need an explicit grant.
    permissions: dict[str, bool] = Field(default_factory=lambda: {"digital_write": False})
    action_permissions: dict[str, str] = Field(default_factory=lambda: {"note": "digital_write"})
    max_utterance_chars: int = 1200


class ModulesCfg(BaseModel):
    disabled: list[str] = Field(default_factory=list)


class WorldCfg(BaseModel):
    location_name: str = "the computer I run on"


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class TraceCfg(BaseModel):
    record_transitions: bool = True
    capture_state: bool = True
    redact_pii: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ORGANISM_", env_nested_delimiter="__", extra="ignore",
        toml_file="config/organism.toml",
    )

    data_dir: Path = Path("data")
    seed: int = 7
    run_label: str = "default"
    identity: IdentityCfg = Field(default_factory=IdentityCfg)
    clock: ClockCfg = Field(default_factory=ClockCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    embeddings: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    brainstem: BrainstemCfg = Field(default_factory=BrainstemCfg)
    attention: AttentionCfg = Field(default_factory=AttentionCfg)
    workspace: WorkspaceCfg = Field(default_factory=WorkspaceCfg)
    working_memory: WorkingMemoryCfg = Field(default_factory=WorkingMemoryCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    thought: ThoughtCfg = Field(default_factory=ThoughtCfg)
    imagination: ImaginationCfg = Field(default_factory=ImaginationCfg)
    sleep: SleepCfg = Field(default_factory=SleepCfg)
    vision: VisionCfg = Field(default_factory=VisionCfg)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    speech: SpeechCfg = Field(default_factory=SpeechCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    modules: ModulesCfg = Field(default_factory=ModulesCfg)
    world: WorldCfg = Field(default_factory=WorldCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    trace: TraceCfg = Field(default_factory=TraceCfg)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)

    def enabled(self, module: str) -> bool:
        return module in ESSENTIAL_MODULES or module not in self.modules.disabled


def load_settings(config_path: str | Path | None = None, **overrides) -> Settings:
    path = Path(config_path or os.environ.get("ORGANISM_CONFIG", "config/organism.toml"))

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(**{**Settings.model_config, "toml_file": path})

    return _FileSettings(**overrides)


def offline_settings(data_dir: str | Path, **overrides) -> Settings:
    """Deterministic, offline configuration: virtual clock, symbolic language fallback,
    hashing embeddings, no hardware sensors."""
    base = dict(
        data_dir=Path(data_dir),
        clock=ClockCfg(mode="virtual"),
        llm=LLMCfg(provider="rule"),
        embeddings=EmbeddingCfg(provider="hash"),
        vision=VisionCfg(enabled=False, detector="none"),
        audio=AudioCfg(microphone=False),
        speech=SpeechCfg(max_per_minute=60),
    )
    base.update(overrides)
    return load_settings(config_path=Path(data_dir) / "__none__.toml", **base)
