# Contributing to CortexAI

Thanks for your interest. CortexAI is a research instrument as much as a program: every capability
is meant to be **measurable** and **ablatable**. The guidelines below keep it that way.

## Getting started

```bash
git clone https://github.com/AdilEddarif/Cortex-AI.git
cd Cortex-AI
pip install -e .[dev]

python -m pytest                 # offline and deterministic
python -m organism benchmark     # 9 conditions x 10 seeds x 12 tasks, about 30 s
python -m organism demo          # the scripted tour
python -m organism chat --thoughts   # talk to it and watch its inner thoughts
```

No GPU or language model is needed: without Ollama the cortex runs fully symbolic and
reproducible (`--llm rule`). Optional senses and voices are extras (`.[vision]`, `.[audio]`, `.[tts]`).

## Design invariants

Changes are reviewed against these. A pull request that breaks one needs a very good reason and a
discussion first.

1. **The language model has two jobs, and both return data.** *Book knowledge* (the `knowledge`
   module and its research agent: impersonal questions only, answered from what it read or fetched)
   and *translation* (the `comprehension` module: an utterance into a fixed structure). Thought,
   deliberation, speech, the self-narrative, dreams and consolidation stay in the cortex's own
   modules. Never route speech, thoughts or self-description through an LLM prompt, never let it
   return free text that is spoken, and never send anything personal to a web service.
2. **Modules communicate only through the bus** (events, requests, nominations). A module never
   reads or writes another module's state. Importing *pure helper functions* (no state, no side
   effects, e.g. `temporal_memory`, `language_rules`) is fine.
3. **Verbal reports never write internal state.** Reports are derived from state snapshots and no
   module consumes them (`tests/test_organism.py::test_verbal_reports_cannot_write_internal_state`).
4. **Honesty over fluency.** The cortex says only what its state, memory, perception or reading
   supports, and says "I don't know" or "I don't remember" otherwise. No confabulation.
5. **Reproducible by default.** Tests and benchmarks use the virtual clock, a fixed seed and the
   symbolic layer, and never touch the network (web lookups are tested against a simulated
   network, see `tests/test_research.py`; `CORTEX_LIVE=1` runs `tests/test_live_2026.py` against the
   real internet). Every model call has a deterministic fallback.
6. **Graceful degradation.** A missing module, model or sensor reduces capability; it never
   crashes cognition. Ablating a module must remove its capability and nothing else.

## Making a change

- **Bug fix:** add a test that fails before the fix. If the bug shows up in conversation, reproduce
  it with `org.converse(...)` in a test (see `tests/test_organism.py`).
- **New behaviour or module:** add a test *and* a benchmark task in
  `organism/experiments/benchmark.py` that states the claim it tests (the task's docstring). If it
  belongs to a module, check that the matching ablation fails that task and that the full system
  passes it.
- **Changes that affect scores:** rerun `python -m organism benchmark --publish` and commit the
  updated `docs/`. If the demo output changes, regenerate `docs/demo.md` and `docs/demo.gif`
  (`python -m organism demo --transcript docs/demo.md` and `python scripts/render_demo_gif.py`).
- **Docs:** update `ARCHITECTURE.md` when you change how a module works, and keep the Mermaid
  diagrams valid.

### Style

Match the surrounding code: type hints, short docstrings that explain *why*, comments only where
the reason is not obvious, and lines up to about 120 characters. Keep replies in
`language_rules.py` short and natural, in the first person, speaking to the person as "you".
Don't add dependencies to the core without discussing it first; optional back-ends go in extras.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one change.
3. Before opening it, check that:
   - [ ] `python -m pytest` passes;
   - [ ] `python -m organism benchmark` shows no unexpected drop for the full system;
   - [ ] new behaviour has a test (and a benchmark task, if it is a capability);
   - [ ] no secrets, personal data, local paths or machine details are committed
     (`data/` and `.env` are git-ignored for this reason).
4. Describe what changed, why, and how you verified it (paste benchmark numbers if relevant).

## Reporting bugs and odd behaviour

Open an issue with the exchange that went wrong. The most useful reports include:

- the conversation, copied from `python -m organism chat --thoughts` (the grey "(thinks)" lines
  show its internal decisions) or from the face UI (the 💭 lines);
- whether a language model was running (`--llm rule` or Ollama with which model);
- what you expected it to say or do.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE)
of this project.
