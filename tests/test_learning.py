"""Understanding beyond the rules, and plasticity: learned skills and learned attention."""
import json

from organism.models.llm import Provider
from organism.modules.comprehension import TEACH, Comprehension, Understanding
from tests.conftest import make_organism, run


class ScriptedModel(Provider):
    """A language model that returns one fixed structure per call (the comprehension schema)."""

    name = "scripted:model"

    def __init__(self, *patches: dict):
        self.patches = list(patches)
        self.prompts: list[str] = []

    async def chat(self, system, prompt, *, schema, images, max_tokens, temperature, vision):
        if "Structured request" not in prompt:          # only the comprehension calls are scripted
            return json.dumps({})                        # anything else falls back to the symbolic layer
        self.prompts.append(prompt)
        return json.dumps(self.patches.pop(0) if self.patches else {"wants": "unclear"})


def test_a_request_to_perform_is_told_from_a_question_about_it(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        told = await org.converse("tell me a joke") or ""
        assert told.endswith(("field.", "everything.", "Fsh.")) and ":" in told     # a piece it has read
        again = await org.converse("tell me another joke") or ""
        assert again != told                                                        # not the same one twice
        riddle = await org.converse("tell me a riddle") or ""
        assert "?" in riddle and riddle != told
        about = await org.converse("what is a joke?") or ""
        assert "Here's one" not in about                                            # a question, not a request
        await org.stop()

    run(scenario())


def test_it_admits_a_request_it_has_no_skill_for(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        reply = await org.converse("dance for me") or ""
        assert "don't know how" in reply and "dance" in reply
        assert "tell me what you mean" in reply or "tell me what you mean by it" in reply
        assert "Okay!" not in reply
        await org.stop()

    run(scenario())


def test_it_learns_what_a_wording_means_and_keeps_it(tmp_path):
    assert TEACH.match("when I say hop, close your eyes").group("trigger").strip() == "hop"
    assert TEACH.match("from now on, if I say hop then smile").group("action").strip() == "smile"

    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        assert "Nothing yet" in (await org.converse("what have you learned?") or "")
        taught = await org.converse("when I say hop, close your eyes") or ""
        assert 'when you say "hop"' in taught and "close my eyes" in taught      # echoed in the first person
        await org.converse("hop")
        await org.tick(1)
        assert org.modules["expression"].view()["name"] == "eyes_closed"          # it really did it
        learned = await org.converse("what have you learned?") or ""
        assert "hop" in learned and "close my eyes" in learned
        await org.stop()

        org = await make_organism(tmp_path)                                       # ... and after a restart
        await org.tick(2)
        await org.converse("hop")
        await org.tick(1)
        assert org.modules["expression"].view()["name"] == "eyes_closed"
        await org.stop()

    run(scenario())


def test_the_model_only_translates_what_the_rules_could_not_place(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        model = ScriptedModel({"wants": "perform", "what": "joke", "confidence": 0.9},
                              {"wants": "act", "action": "sleep", "confidence": 0.9},
                              {"wants": "perform", "what": "backflip", "confidence": 0.9})
        org.ctx.llm.provider = model
        await org.tick(2)
        joke = await org.converse("coud you tell mea jok") or ""                   # a typo, understood
        assert ":" in joke and len(joke) > 30
        assert "goodnight" in (await org.converse("time for bed i think") or "").lower()
        assert "don't know how" in (await org.converse("do a backflip") or "")
        n = len(model.prompts)
        await org.converse("smile")                                                # a form the rules know
        await org.converse("hello")
        assert len(model.prompts) == n                                             # the model is not consulted
        assert set(org.ctx.llm.stats.by_task) <= {"comprehend", "recite", "knowledge_lookup"}
        await org.stop()

    run(scenario())


def test_attention_weights_follow_what_turned_out_to_matter(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        await org.tick(2)
        att = org.modules["attention"]
        before = dict(att.learned)
        for _ in range(3):
            await org.converse("hello there")
            org.hear_sound("a glass shattering", -3.0)
            await org.tick(3)
            org.see_objects(["wall", "floor"])
            await org.tick(3)
        after = dict(att.learned)
        assert att.learning["mattered"] > 0 and att.learning["ignored"] > 0
        assert after != before and after["urgency"] > before["urgency"]        # what led somewhere is reinforced
        assert all(0.05 <= v <= 1.5 for v in after.values())
        await org.stop()

        org = await make_organism(tmp_path)                                    # weights survive a restart
        restored = org.modules["attention"].learned
        assert all(abs(restored[k] - v) < 0.002 for k, v in after.items())
        await org.stop()

    run(scenario())


def test_plasticity_can_be_switched_off(tmp_path):
    async def scenario():
        org = await make_organism(tmp_path)
        org.modules["attention"].cfg.learn = False
        await org.tick(2)
        before = dict(org.modules["attention"].learned)
        for _ in range(3):
            await org.converse("hello there")
            await org.tick(3)
        assert org.modules["attention"].learned == before
        assert org.modules["attention"].learning["updates"] == 0
        await org.stop()

    run(scenario())


def test_comprehension_maps_requests_to_capabilities_it_has():
    u = Understanding.__new__(Understanding)          # the mapping is pure: no bus needed
    u.capabilities = lambda: {"express", "sleep", "recite"}
    assert u._to_patch(Comprehension(wants="perform", what="a joke", confidence=0.9))["command"] == "recite"
    assert u._to_patch(Comprehension(wants="perform", what="poem", topic="rain", confidence=0.9))["command_arg"] \
        == "poem about rain"
    assert u._to_patch(Comprehension(wants="act", action="sleep", confidence=0.9))["command"] == "sleep"
    assert u._to_patch(Comprehension(wants="act", action="fly", what="fly", confidence=0.9))["unsupported"]
    assert u._to_patch(Comprehension(wants="ask_fact", confidence=0.9))["asks_about"] == "general"
    assert u._to_patch(Comprehension(wants="social", confidence=0.9)) == {}
