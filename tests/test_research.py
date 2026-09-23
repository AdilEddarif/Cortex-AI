"""Looking facts up online: Wikipedia + Tavily behind a research agent, on a simulated network."""
import json

import httpx

from organism.models.llm import Provider
from organism.models.research import ResearchAgent
from organism.models.web import WebTools
from tests.conftest import make_organism, run

PAGES = {
    "2026 UEFA Champions League final": "The 2026 UEFA Champions League final was the final match of the 2025–26 "
        "UEFA Champions League. It was played in Budapest on 30 May 2026 between Paris Saint-Germain and Arsenal. "
        "Paris Saint-Germain won the match 4–3 on penalties, following a 1–1 draw after extra time.",
    "Telescope": "A telescope is an optical instrument that uses lenses or mirrors to make distant objects "
                 "appear closer. The first known telescopes were made in the Netherlands in 1608.",
    "Quasar": "A quasar is an extremely luminous active galactic nucleus powered by a supermassive black hole.",
}


class FakeWeb:
    """A simulated internet: Wikipedia search/summary and Tavily search. Records every request."""

    def __init__(self, fail: bool = False):
        self.requests: list[str] = []
        self.fail = fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if self.fail:
            raise httpx.ConnectError("network down")
        if request.url.host == "www.wikidata.org":
            return httpx.Response(200, json=wikidata(request.url.params))
        if request.url.host == "api.tavily.com":
            assert request.headers["Authorization"].startswith("Bearer tvly-")
            q = json.loads(request.content)["query"]
            return httpx.Response(200, json={
                "answer": "The latest Eurovision Song Contest was won by Example Person.",
                "results": [{"title": "France elects a president", "url": "https://www.reuters.com/world/x",
                             "content": f"Coverage of: {q}", "published_date": "2026-05-12"}]})
        if "srsearch" in request.url.params:                                 # Wikipedia search
            q = request.url.params["srsearch"].lower()
            hits = [{"title": t, "snippet": text[:60]} for t, text in PAGES.items()
                    if any(w in q for w in t.lower().split())]
            return httpx.Response(200, json={"query": {"search": hits}})
        title = request.url.params.get("titles", "")                         # an article's introduction
        if title in PAGES:
            return httpx.Response(200, json={"query": {"pages": [{
                "title": title, "extract": PAGES[title], "touched": "2026-08-01T00:00:00Z",
                "fullurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": title, "missing": True}]}})


def wikidata(params) -> dict:
    """A tiny Wikidata: who has held the office of President of the United States."""
    if params.get("action") == "wbsearchentities":
        found = "president of the united states" in params["search"].lower()
        return {"search": [{"id": "Q11696", "label": "President of the United States"}] if found else []}
    ids = params["ids"]
    if ids == "Q11696":
        def holder(qid, start, end, rank="normal"):
            t = lambda d: [{"datavalue": {"value": {"time": f"+{d}T00:00:00Z"}}}]  # noqa: E731
            return {"rank": rank, "mainsnak": {"datavalue": {"value": {"id": qid}}},
                    "qualifiers": {"P580": t(start), "P582": t(end)}}
        return {"entities": {"Q11696": {"labels": {"en": {"value": "President of the United States"}}, "claims": {
            "P1308": [holder("Q6279", "2021-01-20", "2025-01-20"),
                      holder("Q22686", "2025-01-20", "2029-01-20", "preferred")]}}}}   # a scheduled end of term
    names = {"Q6279": {"en": {"value": "Joe Biden"}}, "Q22686": {"mul": {"value": "Donald Trump"}}}  # "mul" only
    return {"entities": {ids: {"labels": names.get(ids, {})}}}


def tools(web: FakeWeb, key: str = "", **kw) -> WebTools:
    return WebTools(transport=httpx.MockTransport(web), tavily_key=key, **kw)


async def cortex(tmp_path, web: FakeWeb, allowed: bool = True, key: str = ""):
    org = await make_organism(tmp_path)
    org.settings.safety.permissions["internet_read"] = allowed
    org.modules["knowledge"].set_tools(tools(web, key))
    await org.tick(2)
    return org


def test_looks_things_up_only_with_permission_and_says_where_from(tmp_path):
    async def scenario():
        web = FakeWeb()
        org = await cortex(tmp_path / "off", web, allowed=False)
        reply = await org.converse("What is a telescope?") or ""
        assert "looked it up" not in reply and web.requests == []        # off by default: no network at all
        await org.stop()

        org = await cortex(tmp_path / "on", web)
        reply = await org.converse("What is a telescope?") or ""
        assert reply.startswith("I looked it up (Wikipedia): A telescope is an optical instrument")
        facts = [m for m in org.modules["memory"].system.records.values() if m.source.startswith("web")]
        assert facts and facts[0].context["sources"][0]["url"].endswith("/wiki/Telescope")
        await org.stop()

    run(scenario())


def test_nothing_personal_or_situational_ever_leaves_the_machine(tmp_path):
    async def scenario():
        web = FakeWeb()
        org = await cortex(tmp_path, web, key="tvly-test-key-123456")
        for q in ["What is my dog's name?", "What is in front of you?", "What's the weather like today?",
                  "Why did you look down?", "What did I tell you about my sister?"]:
            await org.converse(q)
        assert web.requests == []
        await org.stop()

    run(scenario())


def test_time_sensitive_questions_prefer_current_web_search(tmp_path):
    async def scenario():
        web = FakeWeb()
        org = await cortex(tmp_path, web, key="tvly-test-key-123456")
        reply = await org.converse("Who won the latest Eurovision Song Contest?") or ""
        assert "looked it up (reuters.com)" in reply and "Example Person" in reply
        assert web.requests and "tavily" in web.requests[0]                # asked the current web first
        await org.stop()

    run(scenario())


def test_without_a_tavily_key_or_a_network_it_just_does_not_know(tmp_path):
    async def scenario():
        web = FakeWeb()
        org = await cortex(tmp_path / "nokey", web)
        reply = await org.converse("Who is the current president of France?") or ""
        assert "looked it up" not in reply and not any("tavily" in u for u in web.requests)
        await org.stop()

        down = FakeWeb(fail=True)
        org = await cortex(tmp_path / "down", down)
        reply = await org.converse("What is a quasar?") or ""
        assert reply and "looked it up" not in reply and down.requests      # tried, failed, carried on
        await org.stop()

    run(scenario())


def test_budget_and_cache_protect_the_free_credits():
    async def scenario():
        web = FakeWeb()
        t = tools(web, max_calls_per_hour=2)
        assert (await t.wikipedia_page("Telescope"))["title"] == "Telescope"
        assert (await t.wikipedia_page("Telescope"))["title"] == "Telescope"   # cached: no new request
        await t.wikipedia_page("Quasar")
        assert await t.wikipedia_page("Moon") is None                          # over the hourly budget
        assert len(web.requests) == 2 and t.stats["budget_refusals"] == 1
        await t.close()

    run(scenario())


class ScriptedModel(Provider):
    """A language model that plays a scripted sequence of research steps."""

    name = "scripted:model"

    def __init__(self, steps: list[dict]):
        self.steps = list(steps)
        self.prompts: list[str] = []

    async def chat(self, system, prompt, *, schema, images, max_tokens, temperature, vision):
        self.prompts.append(prompt)
        return json.dumps(self.steps.pop(0) if self.steps else {"action": "give_up"})


def _router(provider):
    from organism.config import LLMCfg
    from organism.models.llm import LLMRouter
    r = LLMRouter(LLMCfg(provider="rule"))
    r.provider = provider
    return r


def test_the_agent_answers_only_from_what_it_fetched():
    async def scenario():
        web = FakeWeb()
        model = ScriptedModel([
            {"action": "wikipedia_search", "query": "telescope invention"},
            {"action": "wikipedia_page", "query": "Telescope"},
            {"action": "answer", "answer": "The first known telescopes were made in the Netherlands in 1608.",
             "sources": [2], "confidence": 0.8},
        ])
        agent = ResearchAgent(_router(model), tools(web))
        f = await agent.run("When was the telescope invented")
        assert f.known and f.method == "agent" and "1608" in f.answer
        assert f.sources[0]["url"].endswith("/wiki/Telescope")
        assert [s["action"] for s in f.steps] == ["wikipedia_search", "wikipedia_page", "answer"]
        assert "[2] (Wikipedia" in model.prompts[2]                         # it saw its own notes

        # An answer that cites nothing, or talks as a person, is rejected; the fixed pipeline takes over.
        bad = ScriptedModel([{"action": "answer", "answer": "I think telescopes are great.", "sources": []}])
        f = await ResearchAgent(_router(bad), tools(web)).run("What is a telescope")
        assert f.method == "pipeline" and f.answer.startswith("A telescope is an optical instrument")
        assert "I think" not in f.answer

    run(scenario())


def test_citations_written_in_the_text_count_and_are_not_spoken():
    async def scenario():
        model = ScriptedModel([
            {"action": "wikipedia_search", "query": "telescope"},
            {"action": "answer", "answer": "The first known telescopes were made in 1608.\nThis comes from note [2]."},
        ])
        f = await ResearchAgent(_router(model), tools(FakeWeb())).run("When was the telescope invented")
        assert f.known and f.method == "agent"
        assert f.answer == "The first known telescopes were made in 1608."          # no "[2]" or "note" sentence
        assert f.sources and f.sources[0]["url"].endswith("/wiki/Telescope")

    run(scenario())


def test_it_answers_the_question_asked_or_says_it_did_not_find_it(tmp_path):
    from organism.models.research import _relevant, answer_sentence, search_terms
    assert search_terms("Who won the 2022 FIFA World Cup?") == "2022 FIFA World Cup"
    assert not _relevant("Who won the 2022 FIFA World Cup?", "2026 FIFA U-20 Women's World Cup", "held in 2026")
    intro = PAGES["2026 UEFA Champions League final"]
    assert answer_sentence("who won the champions league title 2026 ?", intro).startswith("Paris Saint-Germain won")
    assert "30 May 2026" in answer_sentence("When was the 2026 Champions League final played?", intro)
    assert "Budapest" in answer_sentence("Where was the 2026 Champions League final played?", intro)
    assert answer_sentence("Who won the 2026 Champions League?", "The final was held in 2026. It was exciting.") is None

    async def scenario():
        org = await cortex(tmp_path / "on", FakeWeb())
        reply = await org.converse("who won the champions league title 2026 ?") or ""
        assert reply.startswith("I looked it up (Wikipedia): Paris Saint-Germain won the match")
        await org.stop()
        org = await cortex(tmp_path / "off", FakeWeb(), allowed=False)
        reply = await org.converse("who won the champions league title 2026 ?") or ""
        assert "internet access is switched off" in reply or "not allowed online" in reply
        await org.stop()

    run(scenario())


def test_it_can_give_up_internet_access_but_never_grant_it_to_itself(tmp_path):
    async def scenario():
        org = await cortex(tmp_path, FakeWeb(), allowed=False)
        reply = await org.converse("turn on internet acess") or ""
        assert "can't switch that on myself" in reply and "/allow internet_read" in reply
        assert org.settings.safety.permissions["internet_read"] is False       # speech never grants
        assert "switched off" in (await org.converse("are you online?") or "")
        org.command("grant", permission="internet_read")                         # the operator's channel
        await org.settle()
        assert org.settings.safety.permissions["internet_read"] is True
        assert (await org.converse("are you online?") or "").startswith("Yes")
        reply = await org.converse("please turn off your internet access") or ""
        assert "I've switched my internet access off" in reply
        assert org.settings.safety.permissions["internet_read"] is False       # speech can give it up
        assert "already off" in (await org.converse("turn off the internet") or "")
        await org.stop()

    run(scenario())


def test_office_holders_come_from_wikidata_with_dates(tmp_path):
    from organism.models.research import office_question
    assert office_question("who is the president of usa 2026 ?") == ("President of the United States", None)
    assert office_question("Who was the president of the United States in 2022?") == ("President of the United States", 2022)
    assert office_question("Who is the current Secretary-General of the United Nations?")[0] ==         "Secretary-General of the United Nations"
    assert office_question("Who won the 2026 FIFA World Cup?") is None

    async def scenario():
        org = await cortex(tmp_path, FakeWeb())
        reply = await org.converse("who is the president of usa 2026 ?") or ""
        assert reply == ("I looked it up (Wikidata): Donald Trump is the President of the United States, "
                         "since 20 January 2025.")
        reply = await org.converse("Who was the president of the United States in 2022?") or ""
        assert "In 2022, the President of the United States was Joe Biden" in reply
        await org.stop()

    run(scenario())
