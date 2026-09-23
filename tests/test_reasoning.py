"""Reasoning: chains of lookups over its own sources, and explaining its own behaviour."""
import httpx

from organism.models.web import WebTools
from organism.modules.reasoning import Reasoning, date_in, number_in, place_in
from tests.conftest import make_organism, run

PAGES = {
    "Eiffel Tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall "
                    "and was completed in 1889.",
    "Empire State Building": "The Empire State Building is a skyscraper in New York City, United States. "
                             "It is 443 metres tall including its antenna.",
    "Christmas": "Christmas is an annual festival held on 25 December 2026 in most of the world.",
}
WIKIDATA = {"President of France": ("Q191954", "Emmanuel Macron", "2017-05-14")}


def fake_internet(request: httpx.Request) -> httpx.Response:
    if request.url.host == "www.wikidata.org":
        params = request.url.params
        if params.get("action") == "wbsearchentities":
            hit = WIKIDATA.get(params["search"])
            return httpx.Response(200, json={"search": [{"id": hit[0], "label": params["search"]}] if hit else []})
        if params["ids"] == "Q191954":
            return httpx.Response(200, json={"entities": {"Q191954": {
                "labels": {"en": {"value": "President of France"}},
                "claims": {"P1308": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q3052772"}}},
                                      "qualifiers": {"P580": [{"datavalue": {"value": {"time": "+2017-05-14T00:00:00Z"}}}]}}]}}}})
        return httpx.Response(200, json={"entities": {"Q3052772": {"labels": {"en": {"value": "Emmanuel Macron"}}}}})
    params = request.url.params
    if "srsearch" in params:
        q = params["srsearch"].lower()
        return httpx.Response(200, json={"query": {"search": [
            {"title": t, "snippet": text[:60]} for t, text in PAGES.items()
            if any(w in q for w in t.lower().split())]}})
    title = params.get("titles", "")
    if title in PAGES:
        return httpx.Response(200, json={"query": {"pages": [{
            "title": title, "extract": PAGES[title], "touched": "2026-09-01T00:00:00Z",
            "fullurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}]}})
    return httpx.Response(200, json={"query": {"pages": [{"title": title, "missing": True}]}})


async def cortex(tmp_path):
    org = await make_organism(tmp_path)
    org.settings.safety.permissions["internet_read"] = True
    org.modules["knowledge"].set_tools(WebTools(transport=httpx.MockTransport(fake_internet)))
    await org.tick(2)
    return org


def test_it_reads_numbers_dates_and_places_out_of_a_sentence():
    assert number_in("It is 330 metres tall.") == (330.0, "metres")
    assert number_in("It was completed in 1889.") is None            # a year is not a measurement
    assert number_in("It is 1,454 feet tall.")[0] > 400
    assert date_in("held on 25 December 2026") == __import__("datetime").date(2026, 12, 25)
    assert place_in("a tower in Paris, France.") == "France"


def test_shapes_are_recognised_and_ordinary_questions_are_left_alone():
    assert Reasoning.shape_of("who is the president of the country where the Eiffel Tower is?") == "multi_hop"
    assert Reasoning.shape_of("which is taller, the Eiffel Tower or the Empire State Building?") == "comparison"
    assert Reasoning.shape_of("how many days until Christmas?") == "time_until"
    assert Reasoning.shape_of("why did you look down?") == "why"
    for ordinary in ("what is a telescope?", "what is my name?", "how are you?", "tell me a joke"):
        assert Reasoning.shape_of(ordinary) is None


def test_a_multi_hop_question_is_worked_out_in_steps(tmp_path):
    async def scenario():
        org = await cortex(tmp_path)
        reply = await org.converse("who is the president of the country where the Eiffel Tower is?") or ""
        assert "Emmanuel Macron" in reply and "I worked it out" in reply and "France" in reply
        steps = org.modules["reasoning"].recent[0]["steps"]
        assert [s["step"] for s in steps][:2] == ["where is the Eiffel Tower", "the country"]
        await org.stop()

    run(scenario())


def test_a_comparison_looks_both_up_and_compares(tmp_path):
    async def scenario():
        org = await cortex(tmp_path)
        reply = await org.converse("which is taller, the Eiffel Tower or the Empire State Building?") or ""
        assert "Empire State Building is taller" in reply and "330 metres" in reply
        await org.stop()

    run(scenario())


def test_time_arithmetic_uses_its_own_clock(tmp_path):
    async def scenario():
        org = await cortex(tmp_path)
        reply = await org.converse("how many days until Christmas?") or ""
        assert "2026-12-25" in reply and "days" in reply
        await org.stop()

    run(scenario())


def test_it_explains_its_own_behaviour_from_what_actually_happened(tmp_path):
    async def scenario():
        org = await cortex(tmp_path)
        await org.converse("Look down.")
        await org.tick(1)
        why = await org.converse("why did you look down?") or ""
        assert why and "asked" in why.lower()
        steps = org.modules["reasoning"].recent[0]["steps"]
        assert steps and any("express" in s["result"] or "look" in s["result"].lower() for s in steps)
        await org.stop()

    run(scenario())


def test_a_broken_chain_is_admitted_not_filled_in(tmp_path):
    async def scenario():
        org = await cortex(tmp_path)
        reply = await org.converse("which is taller, the Eiffel Tower or the Flarnak Monument?") or ""
        assert "couldn't find" in reply.lower() or "could not" in reply.lower()
        assert "Flarnak" in reply and "taller than" not in reply
        assert org.modules["reasoning"].stats["failed"] >= 1
        await org.stop()

    run(scenario())
