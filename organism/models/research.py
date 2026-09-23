"""The research agent: a bounded tool-using loop that looks facts up on the web for the cortex.

It is a delegate, not a second brain. The knowledge module hands it one impersonal question; it
sees nothing of the conversation, the organism or the person. Each step it chooses a tool
(Wikipedia search, a Wikipedia page, a Tavily web search) or gives an answer, and it may only answer
from the notes it fetched, citing them. Answers that talk as a person, cite nothing or cite notes
that do not exist are rejected. Without a language model (or if the model fails) a deterministic
pipeline runs instead: Wikipedia first, Tavily's direct answer as a fallback, current web search
first for time-sensitive questions.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..core.util import content_words, truncate
from .web import WebTools

AGENT_SYSTEM = """You are a research assistant. You look facts up with tools and report only what the fetched notes say.
Never use your own memory for facts: if the notes do not contain the answer, search again or give up.
Today is {today}. Tools:
- wikipedia_search: find Wikipedia articles (query = search words); the top article is read automatically
- wikipedia_page: read the summary of one article (query = exact article title from a search)
{web_tool}- answer: give the answer in at most two short, impersonal sentences (never "I" or "you"), with the note numbers it comes from
- give_up: the question cannot be answered from sources
Prefer {prefer} for this question. Stop as soon as the notes answer it. If they do not answer it yet,
read another article from the search results or search differently before giving up."""

_PERSONA = re.compile(r"\b(I|I'm|I've|me|my|you|your)\b")


class Step(BaseModel):
    action: Literal["wikipedia_search", "wikipedia_page", "web_search", "answer", "give_up"]
    query: str = ""
    answer: str = ""
    sources: list[int] = Field(default_factory=list)
    confidence: float = Field(0.6, ge=0, le=1)


@dataclass
class Finding:
    known: bool
    answer: str = ""
    sources: list[dict] = field(default_factory=list)   # {title, url, date, source}
    confidence: float = 0.0
    method: str = "none"                                 # agent | pipeline | none
    steps: list[dict] = field(default_factory=list)


def _sentences(text: str, n: int = 2, max_chars: int = 320) -> str:
    """The first ``n`` sentences (a sentence may end inside quotes: 'world."'), kept short."""
    text = text.strip()
    ends = [m.end() for m in re.finditer(r"[.!?][\"”’)]*(?=\s|$)", text)]
    out = text[:ends[n - 1]] if len(ends) >= n else text
    if len(out) > max_chars and ends and ends[0] <= max_chars:
        out = text[:ends[0]]
    return out.strip()


def _clean_citations(answer: str) -> str:
    """The answer as it will be spoken: without "[4]" markers or sentences about which note it came from."""
    t = re.sub(r"[^.!?\n]*\b(note|notes|source|sources)\b[^.!?\n]*\[\d+\][^.!?\n]*[.!?]?", "", answer, flags=re.I)
    t = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", t)
    return " ".join(t.split()).strip()


_STOP = {"who", "what", "when", "where", "which", "why", "how", "is", "are", "was", "were", "the", "a", "an", "of",
         "did", "does", "do", "won", "win", "wins", "winner", "tell", "me", "about", "please", "in", "on", "for",
         "to", "and", "many", "much", "current", "currently", "latest", "now", "today"}
_CUES = {
    "won": {"won", "winning", "defeated", "beat", "crowned", "victory", "claimed", "triumphed"},
    "when": {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december", "born", "founded", "released", "launched", "began", "held", "took place"},
    "where": {"located", "held", "hosted", "played", "based", "capital", "city", "country"},
}
# Everyday names for countries and organisations, as they are titled in Wikipedia and Wikidata.
ALIASES = {"usa": "United States", "us": "United States", "u.s.": "United States", "u.s.a.": "United States",
           "america": "United States", "united states of america": "United States", "uk": "United Kingdom",
           "u.k.": "United Kingdom", "britain": "United Kingdom", "great britain": "United Kingdom",
           "uae": "United Arab Emirates", "un": "United Nations", "eu": "European Union", "nato": "NATO"}
_OFFICES = ("president", "prime minister", "chancellor", "king", "queen", "monarch", "emperor", "pope",
            "secretary-general", "secretary general", "vice president", "vice-president", "governor", "mayor",
            "ceo", "chief executive", "director-general", "director general", "head of state", "first minister",
            "taoiseach", "premier", "speaker")
_OFFICE_Q = re.compile(r"\bwho (is|was|are) (the )?(current |present |sitting )?(?P<office>vice president|vice\-president|prime minister|secretary\-general|secretary general|chief executive|director\-general|director general|head of state|first minister|president|chancellor|king|queen|monarch|emperor|pope|governor|mayor|ceo|taoiseach|premier|speaker)(?: of (?P<of>[^?]+))?", re.I)


def office_question(question: str) -> tuple[str, int | None] | None:
    """ "who is the president of usa 2026?" -> ("President of the United States", 2026)."""
    m = _OFFICE_Q.search(question.strip())
    if not m:
        return None
    office, where = m.group("office"), (m.group("of") or "").strip()
    year = next((int(y) for y in _NUM.findall(question)), None)
    where = re.sub(r"\s+(in|during|as of|for)\s*$", "", _NUM.sub("", where).strip(" ?.,"), flags=re.I).strip()
    where = re.sub(r"^the ", "", where, flags=re.I)
    where = ALIASES.get(where.lower(), where)
    if not where:
        return None
    office = office.replace("secretary general", "secretary-general").replace("vice president", "vice-president")
    head = " ".join(w if w in ("of", "the") else w[:1].upper() + w[1:] for w in office.split())
    return (f"{head} of the {where}" if where.startswith("United") or where in ("European Union",) else f"{head} of {where}",
            year if year and year < int(time.strftime("%Y")) else None)


_NUM = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_SENT_END = re.compile(r"[.!?][\"\u201d\u2019)]*(?=\s|$)")


def _expand(words: list[str]) -> list[str]:
    return [ALIASES.get(w.lower(), w) for w in words]


def search_terms(question: str) -> str:
    """What to type into a search box: the question's key terms, years included ("usa" -> "United States")."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2013-]*", question)
    kept = _expand([w for w in words if w.lower() not in _STOP])
    return " ".join(kept) or question


def _long_date(iso: str | None) -> str | None:
    """ "2025-01-20" -> "20 January 2025"."""
    if not iso or len(iso) < 10 or iso[5:7] == "00":
        return iso[:4] if iso else None
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    return f"{int(iso[8:10])} {months[int(iso[5:7]) - 1]} {iso[:4]}" if iso[8:10] != "00" else f"{months[int(iso[5:7]) - 1]} {iso[:4]}"


def rank_hits(question: str, hits: list[dict]) -> list[dict]:
    """Search results in the order worth reading: the question's years standing alone in the title
    ("2026 UEFA Champions League final") before a range that starts with them ("2026–27 ... League")."""
    years = _NUM.findall(question)
    keys = {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2013-]*", question) if w.lower() not in _STOP}

    def score(i_hit: tuple[int, dict]) -> tuple[float, int]:
        i, h = i_hit
        title = h["title"]
        low = title.lower()
        s = 0.0
        for y in years:
            if re.search(r"(?<![\d\u2013-])Y(?![\d\u2013-])".replace("Y", y), title):
                s += 3
            elif y in title:
                s += 1
        s += 0.5 * sum(k in low for k in keys)
        return (-s, i)  # higher score first, then the search engine's own order
    return [h for _, h in sorted(enumerate(hits), key=score)]


def question_type(question: str) -> str | None:
    q = question.lower()
    if re.search(r"\b(who|which (team|club|country|player))\b.*\b(won|win|wins|winner|champion)", q):
        return "won"
    if q.startswith(("when", "what year", "what date")):
        return "when"
    if q.startswith(("where", "in which country", "in which city")):
        return "where"
    if q.startswith(("how many", "how much", "how old", "how tall", "how long")):
        return "number"
    if q.startswith("who"):
        return "who"
    return None


def _relevant(question: str, title: str, text: str) -> bool:
    """The article is about the question: its years all appear, and most of its key words do."""
    hay = (title + " " + text[:1500]).lower()
    if any(n not in hay for n in _NUM.findall(question)):
        return False
    words = [w.lower() for w in _expand(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2013-]*", question)) if w.lower() not in _STOP
             and not _NUM.fullmatch(w)]
    need = len(words) if len(words) <= 2 else (len(words) * 3 + 4) // 5   # few key words: all of them
    return not words or sum(w in hay for w in words) >= need


def answer_sentence(question: str, text: str) -> str | None:
    """The sentence of ``text`` that answers ``question``; None if none does (so nothing is claimed)."""
    sents, start = [], 0
    for m in _SENT_END.finditer(text):
        sents.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text.strip()):
        sents.append(text[start:].strip())
    sents = [s for s in sents if s]
    if not sents:
        return None
    qtype = question_type(question)
    if qtype is None:  # "what is X?": the definition is the first sentence
        return _sentences(text)
    keys = {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2013-]*", question) if w.lower() not in _STOP}
    cues = _CUES.get(qtype, set()) - keys  # 'champions' in 'Champions League' is the topic, not evidence
    best, best_score = None, 0.0
    for s in sents:
        low = s.lower()
        words = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2013-]*", low))
        new_numbers = set(re.findall(r"\d+", s)) - set(re.findall(r"\d+", question))  # the question's year isn't news
        evidence = len(words & cues) + (qtype in ("number", "when") and bool(new_numbers))
        if qtype == "where":  # a place phrase: "in Budapest", "at the Louvre"
            evidence += bool(re.search(r"\b(in|at) (the )?[A-Z][a-z]+", s))
        if qtype == "who":  # a person's name, not a phrase from the question ("United Nations")
            evidence += any(not ({w.lower() for w in n.split()} & keys) for n in re.findall(r"(?<!^)\b[A-Z][a-z]+ [A-Z][a-z]+", s))
        if qtype == "won" and re.search(r"\bwill\b", low):  # "the winners will qualify" is not a result
            continue
        if not evidence:
            continue
        score = 3 * evidence + len(keys & words)  # evidence of an answer outweighs mere topic overlap
        if score > best_score:
            best, best_score = s, score
    if best and len(best) > 200:  # keep the clause that answers: cut at the last comma before 200 characters
        cut = best[:200].rfind(",")
        best = best[:cut].rstrip() + "." if cut > 80 else truncate(best, 200)
    return best


class ResearchAgent:
    def __init__(self, llm, tools: WebTools, max_steps: int = 4):
        self.llm, self.tools, self.max_steps = llm, tools, max_steps

    async def run(self, question: str, time_sensitive: bool = False) -> Finding:
        office = office_question(question)
        if office:  # who holds an office: structured data (Wikidata) beats reading prose
            found = await self._office(*office)
            if found.known:
                return found
        steps: list[dict] = []
        if not self.llm.is_symbolic:
            found = await self._agent(question, time_sensitive)
            if found.known:
                return found
            steps = found.steps
        fallback = await self._pipeline(question, time_sensitive)  # cached pages make this cheap
        fallback.steps = steps + fallback.steps
        return fallback

    # ------------------------------------------------------------------ agent loop
    async def _agent(self, question: str, time_sensitive: bool) -> Finding:
        notes: list[dict] = []
        steps: list[dict] = []
        web = self.tools.has_web_search
        system = AGENT_SYSTEM.format(
            today=time.strftime("%Y-%m-%d"),
            web_tool="- web_search: search the current web, for recent events and anything Wikipedia lacks (query = search words)\n"
            if web else "",
            prefer="web_search (it is about recent or current facts)" if time_sensitive and web else "Wikipedia")
        for i in range(self.max_steps):
            prompt = [f"Question: {question}", "Notes:" if notes else "Notes: (none yet)"]
            prompt += [f"[{n['n']}] ({n['source']}{', ' + n['date'] if n.get('date') else ''}) {n['title']}: {n['text']}"
                       for n in notes]
            if steps:
                prompt.append("Done so far: " + "; ".join(f"{s['action']}({s.get('query', '')})" for s in steps))
            prompt.append("Next step:")
            res = await self.llm.complete("research_step", system, "\n".join(prompt), schema=Step,
                                          fallback=lambda: Step(action="give_up"), priority=3, max_tokens=220,
                                          temperature=0.1)
            if res.model == "rule":  # the model failed: let the pipeline do it
                return Finding(known=False, method="none", steps=steps)
            st: Step = res.data
            steps.append({"action": st.action, "query": truncate(st.query, 80)})
            if st.action == "answer":
                ids = set(st.sources) | {int(x) for x in re.findall(r"\[(\d+)\]", st.answer)}  # cited in the text too
                cited = [n for n in notes if n["n"] in ids]
                answer = _clean_citations(st.answer)
                if answer and cited and not _PERSONA.search(answer):
                    return Finding(known=True, answer=truncate(answer, 400),
                                   sources=[{k: n[k] for k in ("title", "url", "date", "source")} for n in cited],
                                   confidence=round(min(st.confidence, 0.9), 3), method="agent", steps=steps)
                return Finding(known=False, method="agent", steps=steps)   # uncited or persona: rejected
            if st.action == "give_up":
                return Finding(known=False, method="agent", steps=steps)
            new = await self._use_tool(st.action, st.query or question, notes)
            if not new and i == self.max_steps - 1:
                break
        return Finding(known=False, method="agent", steps=steps)

    async def _use_tool(self, action: str, query: str, notes: list[dict]) -> int:
        added = 0

        def note(source: str, title: str, text: str, url: str = "", date: str = "") -> None:
            nonlocal added
            if text:
                limit = 2000 if source != "Wikipedia search" else 240  # whole introductions, short snippets
                notes.append({"n": len(notes) + 1, "source": source, "title": title, "text": truncate(text, limit),
                              "url": url, "date": date})
                added += 1
        if action == "wikipedia_search":
            hits = rank_hits(query, (await self.tools.wikipedia_search(query) or [])[:5])[:3]
            for h in hits:
                note("Wikipedia search", h["title"], h["snippet"])
            if hits:  # read the top article straight away: small models answer better from real text
                p = await self.tools.wikipedia_page(hits[0]["title"])
                if p:
                    note("Wikipedia", p["title"], p["text"], p["url"], p["date"])
        elif action == "wikipedia_page":
            p = await self.tools.wikipedia_page(query)
            if p:
                note("Wikipedia", p["title"], p["text"], p["url"], p["date"])
        elif action == "web_search":
            r = await self.tools.web_search(query)
            if r:
                if r["answer"]:
                    note("web search summary", query, r["answer"])
                for x in r["results"][:3]:
                    note(x["source"], x["title"], x["text"], x["url"], x["date"])
        return added

    async def _office(self, office: str, year: int | None) -> Finding:
        d = await self.tools.wikidata_officeholder(office, year)
        if not d or not d.get("holder"):
            return Finding(known=False, method="none")
        since = _long_date(d.get("start"))
        if year is None:
            answer = f"{d['holder']} is the {d['office']}" + (f", since {since}." if since else ".")
        else:
            answer = f"In {year}, the {d['office']} was {d['holder']}."
        return Finding(known=True, answer=answer, confidence=0.85, method="wikidata",
                       sources=[{"title": d["office"], "url": d["url"], "date": d["date"], "source": "Wikidata"}],
                       steps=[{"action": "wikidata_officeholder", "query": office}])

    # ------------------------------------------------------------------ deterministic pipeline
    async def _pipeline(self, question: str, time_sensitive: bool) -> Finding:
        order = ["web", "wiki"] if time_sensitive else ["wiki", "web"]
        for source in order:
            if source == "wiki":
                hits = rank_hits(question, await self.tools.wikipedia_search(search_terms(question)) or [])
                for h in hits[:3]:
                    p = await self.tools.wikipedia_page(h["title"])
                    if not (p and p.get("text") and _relevant(question, p["title"], p["text"])):
                        continue
                    said = answer_sentence(question, p["text"])
                    if said:
                        return Finding(known=True, answer=said, confidence=0.7, method="pipeline",
                                       sources=[{k: p[k] for k in ("title", "url", "date", "source")}])
            elif self.tools.has_web_search:
                r = await self.tools.web_search(question)
                if r and r["answer"] and not _PERSONA.search(r["answer"]) and _relevant(question, "", r["answer"]):
                    return Finding(known=True, answer=_sentences(r["answer"]), confidence=0.65, method="pipeline",
                                   sources=[{k: x[k] for k in ("title", "url", "date", "source")}
                                            for x in r["results"][:2]])
        return Finding(known=False, method="none")
