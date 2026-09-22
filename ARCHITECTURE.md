# CortexAI: Architecture

> An experimental, brain-inspired cognitive architecture. It does **not** claim to be conscious.
> Consciousness is treated as an open scientific question; the system implements *measurable
> functional properties* (global broadcasting, persistent self-model, temporal continuity,
> self/world distinction, prediction, internal simulation, metacognitive reports) so that they
> can be studied, ablated and compared.

## 1. Design principles

1. **The language model is not the brain.** It plays the role of *book knowledge*: everything a
   well-read person has read. Only the `knowledge` module consults it, for impersonal
   general-knowledge questions, and it returns data (known / answer / confidence). Thought,
   deliberation, speech, the self-narrative, dreams and memory consolidation belong to the
   cortex's own modules.
2. **No all-to-all coupling.** Modules talk only through a single thalamic bus:
   publish/subscribe for events, request/response for queries, nomination for attention.
   No module imports another.
3. **Reproducible by design.** A virtual clock with manual stepping and quiescence detection, a
   fixed seed, and a deterministic symbolic language layer make experiments repeatable.
4. **Graceful degradation.** Every model call has a symbolic fallback and a timeout; a model
   failure never stops cognition.
5. **Reports are not states.** Verbal reports about internal state are separate events that no
   module consumes, so talking about a state can never change it.

## 2. Conceptual loop

```
 sensors ─► perception ─► (nominate) ─► ATTENTION competition ─► GLOBAL WORKSPACE broadcast
   ▲                                                                   │
   │        ┌──────────────┬──────────────┬─────────────┬──────────────┼──────────────┐
   │     memory        thought        emotion        goals        self-model     world-model
   │   (encode/recall) (inner speech) (valuation)  (motivation)  (identity)    (beliefs)
   │        └──────────────┴──────► prediction ◄──┴─────────────┴──────────────┘
   │                                   │ errors re-enter attention
   │                           ACTION SELECTION ◄── imagination (simulate options)
   │                                   │
   │                         SAFETY / POLICY layer
   │                                   │
   └──── external world ◄── speech / face / look / note / sleep ◄── effectors
```

The **brainstem** sets the rhythm: each `TICK` is one cognitive cycle (attention → workspace →
consumers). The cycle speeds up with arousal, from slow when drowsy to fast when highly alert.
Urgent stimuli trigger an immediate "phasic" tick.

## 3. Components

| Module | Brain inspiration | Kind | Responsibilities |
|---|---|---|---|
| `brainstem` | Reticular activating system, hypothalamus | deterministic | arousal, energy, fatigue, sleep pressure, stress, urgency, novelty; modes AWAKE/DROWSY/ASLEEP/DREAMING; tick rhythm; thalamic sensory gain; interoceptive signals |
| bus | Thalamus | deterministic | routing, provenance (`caused_by`), timestamps, sensory gating, request/response, event log, quiescence tracking |
| `attention` | Salience network, pulvinar | algorithm | competition over salience, novelty, urgency, goal relevance, emotional relevance, uncertainty and prediction error; weights modulated by curiosity/fear; an arousal-, energy- and mode-dependent ignition threshold |
| `workspace` | Global neuronal workspace | deterministic | capacity-limited (4) broadcast contents with decay and displacement; a local relay replaces it in ablations |
| `working_memory` | Prefrontal / phonological loop | deterministic | 7 slots, exponential decay, rehearsal, eviction by activation × importance; dialog buffer; unresolved questions |
| `memory` | Hippocampus + cortex | embeddings + database | episodic, semantic, procedural, autobiographical, dream and imagined memories; importance-gated encoding; cue-driven and explicit recall (relevance × retention × importance × strength); temporal memory (see §7) |
| `emotion` | Limbic valuation | deterministic | pleasure, discomfort, curiosity, fear, urgency, social, novelty, frustration, satisfaction, boredom; appraisal + homeostatic decay; causally modulates attention, encoding, arousal and action |
| `goals` | vmPFC / motivational systems | deterministic | persistent intrinsic goals + situational goals (respond, explore surprises); dynamic priorities (rest ∝ fatigue) |
| `thought` | Inner speech / dlPFC | symbolic | bounded thought chains, loop detection, interruption, rate limits, mind-wandering when idle (which may bring back something it has read); think-before-speaking deliberation |
| `language` | Wernicke / Broca | parser + composer | structured understanding (intents, facts, commands, multi-step requests); replies composed from the cortex's actual state in its own words; general-knowledge answers woven in ("From what I've read, ..."); verbal reports logged beside their state snapshots |
| `knowledge` | Semantic cortex fed by reading | language model | answers impersonal questions as data; refuses anything about itself, the speaker or the present moment before the model is consulted; discards answers where a persona leaks in; remembers what it read as semantic memory |
| `speech` | Motor speech | effector | executes approved speech; safety text filter; optional synthesised voice; efference copy |
| `expression` | Facial motor system | effector | timed facial expressions (smile, laugh, wink, closed eyes, looking left/right/up/down, ...) on top of the emotion-driven baseline; facial feedback into valuation |
| `action` | Basal ganglia | algorithm | candidate generation, utility (base + goals + emotion + habits − energy cost), imagination-backed evaluation, one action per channel (vocal / other), deliberate silence, habit learning, committed multi-step plans ("close your eyes, count to 10, then open them": steps run in order, each through safety, each waiting to be seen or heard) |
| `safety` | Inhibitory control | policy | allow-list, capability check, permissions for external actions, rate limits, no external action during sleep |
| `prediction` | Predictive coding | algorithm | visual scene persistence, conversational expectations (including *missing* replies), speaker tone, action outcomes, own arousal; errors drive attention, curiosity, memory and exploration |
| `imagination` | Default mode / hippocampal simulation | rules | outcome/value/risk of actions without acting; dream recombination; deliberate imagining |
| `sleep` | NREM/REM processes | orchestrator | replay → cluster → abstract → strengthen → fade old episodes to gist → prune; self-narrative rewrite; dream-like simulations while external input is gated |
| `self_model` | Self-referential cortex | deterministic | identity, body (an animated face: eyes, eyebrows, mouth; no limbs), sensors, capabilities, static + learned limitations, current state, activity timeline, temporal continuity, predicted future, attributed perception ("I perceived via my camera ...") |
| `world_model` | Posterior cortex / cognitive map | deterministic | entities with beliefs (value, confidence, source, time, decay); object permanence with growing uncertainty; conflict detection |
| `metacognition` | Anterior PFC | deterministic | confidence, knowledge gaps, memory/perception conflicts, source monitoring, model reliability, current focus, all from real metadata |
| `vision` | Visual cortex | object detection (+ optional scene description) | change detection, detection, spatial relations, structured percepts (no raw pixels leave the module) |
| `audition` | Auditory cortex | signal processing + speech recognition | silence / sound / speech classification, speech recognition, loud-sound startle; text channel as a sensory interface |

All model outputs carry confidence and provenance and are never stored as bare facts.

## 4. Event model

Every message is a typed `Event`:

```
id, type, source, timestamp, cycle, payload (validated per type),
summary, modality, confidence, salience, importance, urgency, emotional_value,
uncertainty, gain (thalamic gating), caused_by (provenance), nominated (attention candidate)
```

Types include PERCEPTION, UTTERANCE_UNDERSTOOD, ATTENTION_CHANGED, WORKSPACE_UPDATED,
MEMORY_RETRIEVED/STORED/REPLAYED/CONSOLIDATED, THOUGHT_GENERATED, GOAL_CREATED/UPDATED,
EMOTION_CHANGED, PREDICTION_MADE/ERROR, SIMULATION_RESULT, ACTION_PROPOSED/SELECTED/APPROVED/
REJECTED/EXECUTED, SPEECH_GENERATED, VERBAL_REPORT, SLEEP_STARTED/ENDED, DREAM_STARTED/ENDED/
CONTENT, SELF_STATE_CHANGED, WORLD_UPDATED/CONFLICT, METACOGNITIVE_REPORT, SYSTEM_BOOT/SHUTDOWN,
MODULE_ERROR, COMMAND. A payload that does not match its type is rejected.

**Communication primitives**: `emit/publish` (events), `nominate` (attention candidates),
`request(topic)` (e.g. `memory.recall`, `self.describe`, `world.describe`, `knowledge.query`),
`broadcast` (workspace contents). Capabilities are discovered from `executor.<action>`
responders, so ablating a module removes its capabilities and the rest of the cortex adapts.

## 5. Internal state vs. verbal claims

* Internal state: numeric, owned by modules (body state, valuation state, ...).
* Verbal report: produced by `language` from a snapshot of that state and published as
  `VERBAL_REPORT {report, state_snapshot}`. No module subscribes to it and it is never nominated;
  a structural test enforces this.
* Experiments score **report/state agreement** (e.g. the reported arousal vs. the actual value).

## 6. Persistence and identity

A durable store keeps the full event log, module transitions (input → outputs, state
before/after, model used, latency, confidence), memories with their embeddings (re-embedded
automatically if the embedder changes), goals, world entities and lifecycle data (identity,
creation time, boot count, last shutdown, uptime, body and valuation state, self-narrative,
learned limitations, habits).

After a restart the cortex knows its boot count, how long it was offline (the boot is attended
and remembered), its memories, its goals and its learned limits. Situational goals are marked
abandoned on restart. Nothing is stored as one giant prompt.

## 7. Temporal memory

Memories live in time the way human memories do. Each part can be switched off for ablations.

* **Forgetting curve.** Retention falls off exponentially with the time since a memory was last
  used: `R = exp(-t / S)`. The stability `S` grows with importance and emotional intensity, and
  is much larger for knowledge (semantic, procedural, autobiographical) than for episodes. An
  unimportant episode is effectively gone within days; a vivid one lasts much longer.
* **Spacing effect.** Every recall resets the curve and raises the stability, more when the
  memory was already half-forgotten than when it was fresh, so memories that keep coming back
  become durable.
* **Accessibility.** A faded memory is not deleted at once: it simply stops coming back on a
  weak cue, and only a strong, specific cue can still reach it.
* **Gist over detail.** During sleep, faded episodes lose their exact words and keep only their
  gist ("you talked to me about the telescope"). Very important memories keep their details
  (flashbulb memories). Weak, faded, never-recalled episodes are eventually pruned.
* **Episodes.** Experience is segmented into episodes at pauses, at sleep and at big surprises.
  Each episode keeps who was there, what it was about (the nouns of what was said, not the
  questions asked about it) and what was done, and is summarised in the cortex's own words.
* **Recall by time.** Questions such as "what did we talk about yesterday evening?", "two hours
  ago", "before you fell asleep" or "when we first met" are answered from the episodes, with
  human time labels ("this morning", "yesterday evening", "on Monday"). When nothing is found,
  it says so rather than guessing.

## 8. Time

A real clock (optionally scaled) or a virtual clock (advances per tick; used for experiments).
The temporal state tracks boot time, elapsed time, downtime, episodes (a new episode after sleep
or a long silence) and sleep cycles. The self-model keeps an activity timeline ("five minutes
ago I was talking with Ada").

## 9. Safety

`cognition → ACTION_SELECTED → safety → ACTION_APPROVED/REJECTED → effectors`. Actions that reach
outside the cortex need explicit permission grants. Credentials are never placed in prompts, and
prompts, logs and memories pass through secret redaction (optional PII redaction). The camera
and microphone are off by default, and raw frames are not stored.

## 10. Observability

A live dashboard shows arousal and body state, attention (components, weights, threshold), the
workspace, the current thought and inner monologue, goals, the valuation state with history,
the self-model, the world model, working memory, long-term memory, metacognition, predictions
and errors, action decisions with alternatives and imagined values, safety rejections,
sleep/dreams, per-module load, and a filterable timeline. Selecting an event shows its
transition records.

### Embodied interface (the face)

The face is an effector/sensor surface, not a module: its eyes stream camera frames to `vision`
(tracking on every frame, publishing percepts only on stable scene changes); its ears send
recognised speech to the auditory channel; its mouth renders speech; its expression follows the
brainstem, the valuation state and requested expressions. Nothing the face displays writes back
into the cortex's state.

## 11. Experiments

Each condition runs on a fresh cortex with the same protocol: introductions → perception →
probes → a simultaneous audio-visual event → distraction → a delay → probes → restart after
time offline → probes.

Conditions: baseline, without long-term memory, without the self-model, without the global
workspace, sensory deprivation, and single-module lesions.

Measured: continuity, identity, introspective accuracy, perception, cross-modal integration,
self-knowledge, self-reference rate, response rate, workspace modality mix, share of decisions
made with multimodal context, and share of imagined decisions. Sensory deprivation measures
internal activity (thoughts, replays, dreams, abstractions) with no input.
