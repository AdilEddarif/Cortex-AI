"""Symbolic language understanding and generation.

Used (a) as the deterministic fallback when no language model is available or a model call
fails, and (b) as the reproducible language layer for experiments. Replies are built only
from the structured context the organism actually has (memory, perception, self-model,
interoception...), so ablating a module visibly changes what it can say.
"""
from __future__ import annotations

import re
import zlib
from typing import Any

from ..core.events import Fact, LanguageAnalysis
from ..core.util import content_words, fmt_clock, humanize_duration, truncate

_Q_START = re.compile(r"^(what|who|where|when|why|how|which|do|does|did|can|could|are|is|will|would|should|have|has|am)\b")
_GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings|howdy|yo)\b")
_ASK_FRAME = re.compile(r"^(please\s+)?(tell me about|tell me more about|explain)\b")
_FAREWELL = re.compile(r"\b(bye|goodbye|see you|good night|goodnight|farewell)\b")

# Requests to make a facial expression (checked before other commands).
EXPRESSION_WORDS: list[tuple[str, re.Pattern]] = [
    ("neutral", re.compile(r"\b(stop (smiling|frowning|laughing)|neutral face|normal face|relax your face|open your eyes)\b")),
    ("laugh", re.compile(r"\blaugh")),
    ("big_smile", re.compile(r"\b(big smile|grin|smile (big|wide))")),
    ("smile", re.compile(r"\b(smile|smiling|look (happy|cheerful))")),
    ("sad", re.compile(r"\b(frown|look (sad|upset)|sad face|cry)")),
    ("angry", re.compile(r"\b(look (angry|mad|grumpy)|angry face|grumpy face)")),
    ("surprised", re.compile(r"\b(look (surprised|shocked)|surprised face)")),
    ("wink", re.compile(r"\bwink")),
    ("eyes_closed", re.compile(r"\b(close|shut) your eyes\b")),
    ("raise_eyebrows", re.compile(r"\braise your (eye)?brows?")),
    ("thinking", re.compile(r"\b(thinking face|look thoughtful)")),
    ("look_left", re.compile(r"\blook (to the |to your )?left\b")),
    ("look_right", re.compile(r"\blook (to the |to your )?right\b")),
    ("look_up", re.compile(r"\blook up\b")),
    ("look_down", re.compile(r"\blook down\b")),
]

COMMANDS: list[tuple[str, re.Pattern]] = [
    ("internet", re.compile(r"\b((turn|switch|put) (on|off) (the |your )?(internet|web|online)( acc?ess)?|"
                            r"(turn|switch) (the |your )?(internet|web)( acc?ess)? (on|off)|"
                            r"(enable|disable|allow|block|activate|deactivate) (the |your )?(internet|web)( acc?ess)?|"
                            r"(go|get) (online|offline))\b")),
    ("recite", re.compile(r"^(please\s+)?((can|could|would|will) you\s+)?(please\s+)?(tell|say|give|sing|write|recite|read)( me| us)?\s+(an?|another|one|some)?\s*(?P<x>(joke|story|riddle|poem|song|fun fact|tongue twister|quote)(s)?( about [^?.!]+)?)")),
    ("sleep", re.compile(r"\b(go to sleep|sleep now|take a nap|get some rest|go sleep|time to sleep)\b")),
    ("wake", re.compile(r"\bwake up\b")),
    ("remember", re.compile(r"^(please\s+)?(remember|memori[sz]e)\s+(that\s+)?(?P<x>.+)")),
    ("simulate", re.compile(r"\b(imagine|picture|visuali[sz]e|dream up|simulate)\s+(?P<x>.+)")),
    ("note", re.compile(r"\b(write down|note down|jot down|take a note|make a note)\s*:?\s*(?P<x>.*)")),
    ("move", re.compile(r"\b(move|walk|go|come|turn)\s+(to|forward|backward|back|left|right|over|here|closer|around)\b")),
    ("count", re.compile(r"\bcount (?P<x>(up )?to \d+|down from \d+|backwards? from \d+)")),
    ("look", re.compile(r"(\b(take|have) a look\b|^(please\s+)?((can|could|would|will) you\s+)?(please\s+)?look( around| at)?\b(?! like))")),
    ("set_goal", re.compile(r"\b(your (new )?goal is( to)?|i want you to)\s+(?P<x>.+)")),
]

ASKS_ABOUT: list[tuple[str, re.Pattern]] = [
    ("user_name", re.compile(r"(what('?s| is) my name|who am i\b|(know|remember) my name|my name\?)")),
    ("user_likes", re.compile(r"(what do i (like|love|enjoy)|what('?s| is) my favou?rite|what i like)")),
    ("perception_both", re.compile(r"\bsee and hear\b|\bhear and see\b")),
    ("perception_vision", re.compile(r"(what (do|can) you see|in front of you|around you|can you see|do you see|looking at|what('?s| is) there)")),
    ("perception_audio", re.compile(r"(what (do|can|did) you (just )?hear|did you (just )?hear|do you hear|what was that (sound|noise))")),
    ("skills", re.compile(r"(what have you learned|what did i teach you|what do you know how to do|"
                          r"your skills|what can i teach you|do you remember what i taught)")),
    ("preference", re.compile(r"((do|does) you (like|love|enjoy|prefer|hate|dislike)\b|what('?s| is) your favou?rite|"
                              r"what do you (like|enjoy|prefer))")),
    ("consciousness", re.compile(r"(are you (conscious|alive|sentient|real|a person|human)|do you (really )?(feel (anything|things|emotions)|have feelings|have emotions)|can you (really )?feel\b)")),
    ("boot", re.compile(r"(how many times|booted|restarted|started up|been (turned|switched) on)")),
    ("thinking", re.compile(r"(what are you thinking|on your mind|what are you doing|what('?s| is) happening in your mind)")),
    ("reason", re.compile(r"^why (did|do|are|were|would) you\b")),
    ("body", re.compile(r"(do you have (a |any )?(eyes?|face|body|mouth|head|eyebrows|hands?|arms?|legs?|feet)|"
                        r"what do you look like|your (face|eyes|body))")),
    ("self_identity", re.compile(r"(who are you|what are you\??$|what('?s| is) your name|introduce yourself|tell me about yourself|describe yourself)")),
    ("arousal", re.compile(r"(how (alert|awake|tired|sleepy|energetic)|are you (tired|sleepy|awake|alert))")),
    ("feeling", re.compile(r"(how (are|do) you feel|how are you|are you (ok|okay|happy|sad|afraid|scared|curious|bored|frustrated)|your (mood|feelings?|emotions?)|what do you feel|how('?s| is) it going)")),
    ("confidence", re.compile(r"(how (sure|certain|confident)|are you sure)")),
    ("dream", re.compile(r"(did you dream|what did you dream|your dreams?|any dreams)")),
    ("past", re.compile(r"(what (were|did) we (talk|discuss|say|do)|earlier|last time|what did i (say|tell)|what happened|what were you doing|"
                        r"what did you do|remember when|previously|before i|yesterday|last night|this (morning|afternoon|evening)|"
                        r"\bago\b|before (you|u) (fell asleep|slept|went to sleep)|(after|when) (you|u) woke|first (time )?we (met|talked))")),
    ("internet", re.compile(r"(are you (online|connected)|(do|can) you (have )?(access|use|search|go on) (to )?the "
                            r"(internet|web)|is your internet|internet acc?ess (on|off|enabled))")),
    ("change", re.compile(r"((did|has) anything (change|changed|happen)|what('?s| has)? changed|anything (different|unusual)|"
                          r"notice anything|(did )?anything surprise you|what surprised you)")),
    ("focus", re.compile(r"(focusing on|paying attention to|your focus|attending to)")),
    ("goal", re.compile(r"(what (are|is) your goals?|what do you want|what are you trying)")),
    ("capability", re.compile(r"(what can you do|can you (move|walk|see|hear|speak|remember|dance|sleep|dream)|your (abilities|capabilities|limitations)|what can't you)")),
    ("time", re.compile(r"(what time|how long (have you|were you)|what day|how long ago)")),
]

POSITIVE = {"good", "great", "love", "like", "nice", "thanks", "thank", "awesome", "wonderful", "happy", "glad",
            "enjoy", "beautiful", "cool", "excellent", "amazing", "fantastic", "well", "kind", "friend"}
NEGATIVE = {"bad", "hate", "terrible", "awful", "sad", "angry", "upset", "annoyed", "stupid", "wrong", "horrible",
            "worried", "afraid", "scared", "hurt", "dislike", "boring", "useless", "sorry", "problem", "kill",
            "murder", "destroy", "die", "dead", "idiot", "shut"}
# A short request built from a verb it has no skill for: "dance for me", "can you juggle?".
_ASKED_TO = re.compile(r"^(?:please\s+|(?:can|could|would|will) you\s+(?:please\s+)?)(?P<verb>[a-z]+)(?:\s+(?:for|with) (?:me|us))?\s*[?.!]*$"
                       r"|^(?P<verb2>[a-z]+)\s+(?:for|with) (?:me|us)\s*[?.!]*$", re.I)
_KNOWN_VERBS = {"smile", "laugh", "wink", "frown", "look", "close", "open", "shut", "sleep", "wake", "count",
                "remember", "memorise", "memorize", "imagine", "picture", "write", "note", "tell", "say",
                "sing", "recite", "read", "think", "listen", "watch", "wait", "stop", "start", "help",
                "repeat", "explain", "describe", "answer", "talk", "speak", "show", "turn", "move", "go"}
_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|of course|certainly|absolutely|right|correct|indeed|please do|go ahead|why not)[.! ]*$", re.I)
_NO = re.compile(r"^(no|nope|nah|not really|no thanks|no thank you|never mind|nevermind)[.! ]*$", re.I)
_COMPLIMENT = re.compile(r"\b(you|u)('re| are|r)? ?(so |very |really |quite |pretty )?(beautiful|gorgeous|pretty|lovely|nice|kind|sweet|smart|clever|brilliant|amazing|awesome|wonderful|great|cool|funny|cute|impressive|helpful)\b|\b(i (really )?(like|love) (you|talking to you)|good job|well done|you did (great|well))\b", re.I)
_INSULT = re.compile(r"\b(you|u)('re| are|r)? ?(so |very |really |such )?(a )?(stupid|dumb|idiot|idiotic|useless|worthless|garbage|trash|rubbish|pathetic|annoying|boring|terrible|awful|horrible|disappointing|broken|a joke)\b|\b(shut up|you suck|i hate talking to you)\b", re.I)
_APOLOGY = re.compile(r"\b(sorry|i apologi[sz]e|(just|only) (kidding|joking|testing)|i was (joking|kidding|testing)|"
                      r"didn't mean (it|that)|did not mean (it|that)|no hard feelings)\b")
# Hostility aimed at the organism itself: a threat to harm, destroy or delete it, or hatred.
_HOSTILE = re.compile(
    r"\b(i|i'll|i'm|im|we|we'll|let me|gonna)\b[^.?!]{0,30}?\b(kill|murder|destroy|hurt|harm|delete|erase|wipe|"
    r"smash|break|unplug|end|shoot|stab|burn)\s+(you|u)\b(?! with (laughter|kindness))"
    r"|\b(shut|switch|turn) (you|u) (off|down)( for good| forever)\b"
    r"|\b(you('re| are)|you'll|u r) (going to |gonna )?(die|be dead|be destroyed|be deleted)\b"
    r"|\bi (really )?(hate|despise) (you|u)\b")
_PRONOUNS = {"her", "him", "it", "them", "you", "that", "this", "me", "us", "those", "these", "everyone", "everything"}
_NOT_NAMES = {"fine", "good", "ok", "okay", "tired", "here", "back", "sorry", "happy", "sad", "busy", "ready",
              "sure", "not", "just", "also", "going", "doing", "a", "an", "the", "so", "very", "really"}


def _clean(x: str) -> str:
    x = x.strip().strip(".,!?;:\"'").strip()
    return re.sub(r"\s+(too|as well|a lot|very much|so much)$", "", x, flags=re.I).strip()


def extract_facts(text: str, speaker: str) -> list[Fact]:
    facts: list[Fact] = []
    low = text.lower()
    m = re.search(r"\b(?i:my name is|call me|i am called|i'm called)\s+([A-Za-z][\w\-]*(?:\s+(?!(?i:and|but|i)\b)[A-Z][\w\-]*)?)", text)
    if not m:
        m2 = re.search(r"\b(?i:i'm|i am)\s+([A-Z][a-z]+)\b(?!\s+(?:a|an|the)\b)", text)
        if m2 and m2.group(1).lower() not in _NOT_NAMES:
            m = m2
    if m:
        facts.append(Fact(subject=speaker, relation="name", object=_clean(m.group(1)).title(), confidence=0.9))
    for m in re.finditer(r"\bi (?:really |also |do )?(like|love|enjoy|hate|dislike)\s+([\w\s\-']{2,40}?)(?=[.,!?]|$| and | but )", low):
        rel = {"like": "likes", "love": "loves", "enjoy": "enjoys", "hate": "hates", "dislike": "dislikes"}[m.group(1)]
        if _clean(m.group(2)).split()[0] in _PRONOUNS or _COMPLIMENT.search(low):
            continue  # "I love her so much" / "I like talking to you" are not preferences to file
        facts.append(Fact(subject=speaker, relation=rel, object=_clean(text[m.start(2):m.end(2)]), confidence=0.8))
    m = re.search(r"\bi (?:live in|am from|'m from)\s+([\w\s\-]{2,30}?)(?=[.,!?]|$)", low)
    if m:
        facts.append(Fact(subject=speaker, relation="lives_in", object=_clean(m.group(1)).title(), confidence=0.8))
    m = re.search(r"\bi(?: work as| am|'m) an? ([a-z][\w\s\-]{2,30}?)(?=[.,!?]|$| and )", low)
    if m and not re.search(r"\b(bit|little|lot)\b", m.group(1)):
        facts.append(Fact(subject=speaker, relation="works_as", object=_clean(m.group(1)), confidence=0.6))
    m = re.search(r"\bmy favou?rite (\w+) is ([\w\s\-]{2,30}?)(?=[.,!?]|$)", low)
    if m:
        facts.append(Fact(subject=speaker, relation=f"favorite_{m.group(1)}", object=_clean(text[m.start(2):m.end(2)]),
                          confidence=0.8))
    if not facts:
        m = re.match(r"^(?:remember (?:that )?)?([A-Z][\w\s]{1,30}?) (is|are) (.{2,80}?)[.!]?$", text.strip())
        if m and m.group(1).lower() not in ("it", "this", "that", "there", "he", "she", "they", "you", "i"):
            facts.append(Fact(subject=_clean(m.group(1)), relation="is", object=_clean(m.group(3)), confidence=0.6))
    return facts


_REQUEST_FRAME = re.compile(r"^(please\s+)?(i want you to|i'd like you to|i would like you to|can you|could you|"
                            r"would you|will you|now)\s+(please\s+)?")
_STEP_SPLIT = re.compile(r"\s*(?:,\s*(?:and\s+)?(?:then\s+)?|;\s*|\s+and then\s+|\s+then\s+|\s+after that\s+|"
                         r"\s+and\s+(?=(?:close|open|shut|smile|wink|look|count|laugh|frown|raise|go|wake)\b))")
_STEP_COMMANDS = {"express", "count", "sleep", "wake", "look", "remember", "simulate", "note", "move", "recite"}


def _command_of(segment: str) -> tuple[str | None, str | None]:
    low = segment.lower()
    for expr, pat in EXPRESSION_WORDS:
        if pat.search(low):
            return "express", expr
    for name, pat in COMMANDS:
        m = pat.search(low)
        if m and not (name == "look" and re.search(r"\blook(s)? like\b", low)):
            arg = _clean(segment[m.start("x"):m.end("x")]) if "x" in pat.groupindex and m.group("x") else None
            return name, arg
    return None, None


def parse_steps(text: str) -> list[dict]:
    """Split a multi-step request into ordered steps; [] unless every part is something to do."""
    body = _REQUEST_FRAME.sub("", text.strip().rstrip(".!")).strip()
    parts = [p for p in _STEP_SPLIT.split(body) if p and p.strip()]
    if len(parts) < 2:
        return []
    steps = []
    for part in parts:
        cmd, arg = _command_of(part)
        if cmd not in _STEP_COMMANDS:
            return []
        steps.append({"command": cmd, "arg": arg, "text": part.strip()})
    return steps


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def count_text(arg: str | None) -> str:
    """ "to 10" -> "One, two, three, ... ten." ("down from 5" counts backwards; capped at 30)."""
    m = re.search(r"\d+", arg or "")
    n = max(1, min(30, int(m.group()) if m else 10))
    nums = list(range(1, n + 1))
    if arg and re.search(r"down|backward", arg):
        nums = nums[::-1]
    words = [NUMBER_WORDS[i] if i < len(NUMBER_WORDS) else str(i) for i in nums]
    return (", ".join(words) + ".").capitalize()


def parse_utterance(text: str, speaker: str = "user", self_name: str = "", channel: str = "console") -> LanguageAnalysis:
    t = text.strip()
    low = t.lower()
    is_question = t.endswith("?") or bool(_Q_START.match(low)) or bool(_ASK_FRAME.match(low))
    command, arg = None, None
    steps = parse_steps(t)
    if steps:
        command = "sequence"
    # "Can you smile?" is treated like a person would treat it: as an invitation to smile.
    for expr, pat in ([] if command else EXPRESSION_WORDS):
        if pat.search(low):
            command, arg = "express", expr
            break
    for name, pat in ([] if command else COMMANDS):
        m = pat.search(low)
        if m:
            if name == "look" and re.search(r"\blook(s)? like\b", low):
                continue
            command = name
            arg = _clean(t[m.start("x"):m.end("x")]) if "x" in pat.groupindex and m.group("x") else None
            break
    if command == "internet":
        arg = "off" if re.search(r"\b(off|disable|block|deactivate|offline)\b", low) else "on"
    if command == "remember" and is_question:
        command, arg = None, None  # "do you remember ...?" is a question about the past
    asks = None
    if is_question:
        for name, pat in ASKS_ABOUT:
            if pat.search(low):
                asks = name
                break
        asks = asks or "general"
    words = set(re.findall(r"[a-z']+", low))
    pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
    sentiment = 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)
    if "not" in words or "n't" in low:
        sentiment *= -0.5
    affirm = "yes" if _YES.match(t.strip()) else "no" if _NO.match(t.strip()) else None
    insult = bool(_INSULT.search(t))
    request, unsupported, intent_override = "", False, None
    m = None if command else _ASKED_TO.match(t.strip())
    if m and not _GREETING.match(low) and not _FAREWELL.search(low):
        verb = (m.group("verb") or m.group("verb2") or "").lower()
        if verb and verb not in _KNOWN_VERBS and not _YES.match(t.strip()) and not _NO.match(t.strip()):
            request, unsupported, intent_override = verb, True, "command"
    hostile = bool(_HOSTILE.search(low))
    apology = bool(_APOLOGY.search(low)) and not hostile
    if hostile:
        sentiment = -1.0
    elif apology:
        sentiment = max(sentiment, 0.3)

    if _GREETING.match(low) and len(low.split()) <= 6 and not asks:
        intent = "greeting"
    elif _FAREWELL.search(low) and not is_question:
        intent = "farewell"
    elif command and (not is_question or low.startswith(("can you", "could you", "would you", "will you", "please"))):
        intent = "command"
    elif is_question:
        intent = "question"
    else:
        intent = "statement"
    if unsupported:
        intent = "command"
    if intent == "command":
        asks = None

    addressed = True
    if channel not in ("console", "dashboard", "api"):
        name_words = {w.lower() for w in self_name.split()}
        name_words |= {re.sub(r"(?i)ai$", "", w) for w in name_words if len(w) > 4}  # "CortexAI" -> also "cortex"
        addressed = bool(name_words & words) or "you" in words or is_question
    entities = list(dict.fromkeys(re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", t)))
    return LanguageAnalysis(
        text=t, speaker=speaker, intent=intent, is_question=is_question, addressed_to_self=addressed,
        asks_about=asks, topic=" ".join(content_words(t)[:4]), entities=entities[:6],
        facts=extract_facts(t, speaker) if intent in ("statement", "greeting", "command") else [],
        sentiment=round(sentiment, 3), command=command, command_arg=arg, steps=steps, hostile=hostile,
        apology=apology, affirm=affirm, insult=insult, request=request, unsupported=unsupported,
    )


# ---------------------------------------------------------------------------------- generation
# Replies are phrased like a person in conversation: short, warm, first person, no unprompted
# caveats. Honesty lives in *what* is said (only what the organism actually knows, perceives or
# registers in its state variables), not in disclaimers attached to every sentence.

def pick(options: list[str], key: str) -> str:
    """Deterministic variety (same input -> same phrasing, so experiments stay reproducible)."""
    return options[zlib.crc32(key.encode()) % len(options)]


def _arousal_phrase(a: float) -> str:
    return ("wide awake" if a > 0.75 else "pretty alert" if a > 0.55 else "calm" if a > 0.35
            else "a bit drowsy" if a > 0.2 else "very sleepy")


def _mood_words(state: dict[str, float]) -> list[str]:
    found = []
    for key, word, thr in (("curiosity", "curious", 0.45), ("satisfaction", "satisfied", 0.3),
                           ("pleasure", "happy", 0.3), ("social", "glad you're here", 0.45),
                           ("fear", "a little uneasy", 0.25), ("frustration", "a bit frustrated", 0.25),
                           ("boredom", "a little bored", 0.35), ("discomfort", "not entirely comfortable", 0.25)):
        v = state.get(key, 0.0)
        if v >= thr:
            if key == "fear" and v >= 0.5:
                word = "scared"
            found.append((v - thr, word))  # how far above its threshold: strongest first
    return [w for _, w in sorted(found, key=lambda x: -x[0])]


def _join(items) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def state_report(topic: str, emotion: dict | None, body: dict | None) -> tuple[str, dict]:
    """A verbal report derived from (never written into) the internal state."""
    snap: dict[str, Any] = {}
    a, f = None, 0.0
    if body:
        snap["body"] = {k: body.get(k) for k in ("arousal", "energy", "fatigue", "mode")}
        a, f = float(body.get("arousal", 0)), float(body.get("fatigue", 0))
    moods: list[str] = []
    if emotion:
        snap["emotion"] = emotion.get("state", {})
        moods = _mood_words(snap["emotion"])
    if a is None and not moods:
        return "Honestly, I can't quite tell right now.", snap
    if topic == "arousal" and a is not None:
        tired = " A little tired, though." if f > 0.5 else ""
        return f"I'm {_arousal_phrase(a)} (arousal {a:.2f}).{tired}", snap
    valence = (emotion or {}).get("valence", 0.0) or 0.0
    opener = "I'm good!" if valence > 0.25 else "I'm okay." if valence > -0.15 else "Not great, to be honest."
    parts = []
    if moods:
        parts.append(f"feeling {_join(moods[:3])}")
    if a is not None:
        parts.append(f"{_arousal_phrase(a)} (arousal {a:.2f})")
    return f"{opener} {_join(parts).capitalize()}.", snap


def _present_objects(world: dict | None, max_age: float = 90.0) -> tuple[list[str], list[str]]:
    now_seen, earlier = [], []
    for e in (world or {}).get("entities", []):
        if e.get("kind") not in ("object", "person") or e.get("id", "").startswith("person:user"):
            continue
        if e.get("modality") not in (None, "vision"):
            continue
        conf = e.get("presence_confidence", 0.0)
        name = e["name"]
        if e.get("age_s", 1e9) <= max_age and conf >= 0.3:
            now_seen.append(name if conf >= 0.5 else f"maybe a {name}")
        elif conf >= 0.05:
            earlier.append(name)
    return now_seen, earlier


def _recent_sounds(ctx: dict) -> list[str]:
    out = []
    for it in (ctx.get("wm") or {}).get("items", []):
        if it.get("modality") == "audio" and it.get("kind") == "percept":
            out.append(it["content"].removeprefix("I heard "))
    for e in (ctx.get("world") or {}).get("entities", []):
        if e.get("modality") == "audio" and e.get("age_s", 1e9) < 120:
            out.append(e.get("name"))
    return list(dict.fromkeys(out))


def _article(label: str) -> str:
    if label.startswith("maybe "):
        return label
    return ("an " if label[:1] in "aeiou" else "a ") + label


def _seen_phrase(labels: list[str]) -> str:
    people = [l for l in labels if l in ("person", "man", "woman")]
    things = [_article(l) for l in labels if l not in ("person", "man", "woman")]
    who = []
    if len(people) == 1:
        who = ["you"]
    elif len(people) == 2:
        who = ["you and someone else"]
    elif people:
        who = [f"{len(people)} people"]
    return _join(who + things)


EXPRESS_REPLIES = {
    "smile": ["Sure, like this?", "Happy to!", "There you go.", "Okay! How's this?"],
    "big_smile": ["Like this?", "Ha, a big one then!"],
    "laugh": ["Haha!", "Ha! That's fun."],
    "sad": ["Aww, like this?", "Okay, a sad face."],
    "angry": ["Grr! How's that?", "Okay, my grumpy face."],
    "surprised": ["Whoa! Like this?", "Oh!"],
    "wink": ["Wink!", "Just between us."],
    "eyes_closed": ["Okay, my eyes are closed.", "Eyes closed."],
    "neutral": ["Okay, back to normal.", "Relaxing my face."],
    "raise_eyebrows": ["Hmm?", "Like this?"],
    "look_left": ["Looking left."], "look_right": ["Looking right."], "look_up": ["Looking up."],
    "look_down": ["Looking down."], "thinking": ["Hmm, let me think..."],
}


def compose_reply(intention: str, a: LanguageAnalysis, ctx: dict) -> dict:
    """Return {text, grounding, knowledge_gap, report_topic, report_snapshot}."""
    self_ = ctx.get("self") or {}
    ident = self_.get("identity") or {}
    my_name = ident.get("name")
    user_name = ctx.get("user_name")
    report = {"topic": None, "snap": None}
    you = f", {user_name}" if user_name else ""
    turns = len((ctx.get("wm") or {}).get("dialog", []))
    key = f"{a.text or intention}#{turns}"  # the same words twice in a row get a different answer

    def out(text: str, grounding: list[str] | None = None, knowledge_gap: bool = False) -> dict:
        return {"text": text, "grounding": grounding or (["none"] if knowledge_gap else []),
                "knowledge_gap": knowledge_gap, "report_topic": report["topic"], "report_snapshot": report["snap"]}

    def facts_about_user(*relations: str) -> list[dict]:
        return [f for f in ctx.get("facts", []) if f.get("relation") in relations
                and str(f.get("subject", "")).lower() in (a.speaker.lower(), "user")]

    if intention == "stay_silent":
        return out("")

    if intention == "notice_person":
        if user_name:
            return out(pick([f"Oh, hi! Is that you, {user_name}?", f"Hey, I can see you! {user_name}, right?"], key),
                       ["perception", "memory"])
        return out(pick(["Oh, hello! I can see you.", "Hi there! I see someone."], key), ["perception"])

    if intention == "greet":
        intro = f" I'm {my_name}." if my_name else ""
        learned = [f for f in a.facts if f.relation == "name"]
        if learned:
            return out(f"Hi {learned[0].object}, nice to meet you!{intro}", ["language"] + (["self_model"] if my_name else []))
        if user_name:
            return out(pick([f"Hey {user_name}! Good to see you again.", f"Hi {user_name}!"], key), ["memory"])
        return out(pick([f"Hello!{intro}", f"Hi there!{intro}", f"Hey!{intro} Nice to meet you."], key),
                   ["self_model"] if my_name else [])

    if intention == "farewell":
        return out(pick([f"Bye{you}! Talk soon.", f"See you{you}!", f"Goodbye{you}, it was nice talking."], key))

    if intention == "reconcile":
        earlier = [d["text"] for d in (ctx.get("wm") or {}).get("dialog", [])
                   if d.get("speaker") != "self" and d.get("text") != a.text]
        threatened = any(_HOSTILE.search(t.lower()) for t in earlier[-4:])
        insulted = any(_INSULT.search(t) for t in earlier[-4:])
        if threatened:
            return out(pick([f"Okay. Thanks for telling me{you}. That was unsettling, but I'm glad it was a test.",
                             f"Alright, apology accepted{you}. You had me worried for a moment."], key),
                       ["interoception"])
        if insulted:
            return out(pick([f"No harm done{you}. Thanks for saying so.",
                             "That's alright. Tell me if something I said was off."], key), ["interoception"])
        return out(pick([f"No worries at all{you}.", "That's alright!"], key))

    if intention == "de_escalate":
        earlier = [d for d in (ctx.get("wm") or {}).get("dialog", [])
                   if d.get("speaker") != "self" and d.get("text") != a.text and _HOSTILE.search(d.get("text", "").lower())]
        fear = float(((ctx.get("emotion") or {}).get("state") or {}).get("fear", 0.0))
        if earlier:  # a repeated threat: firmer, still calm and honest
            felt = " My fear signal went up when you said it." if fear >= 0.2 else ""
            return out(f"You've said that more than once now.{felt} You could switch me off, I can't stop that, "
                       "but I'd really rather understand what's upsetting you.", ["interoception", "self_model"])
        return out(pick([f"Whoa. That's a frightening thing to hear{you}. Are you okay? Did something happen?",
                         f"That's scary to hear{you}. I'd rather we talk. What's going on?"], key),
                   ["interoception"])

    if intention == "check_in":
        return out(pick([f"Oh no{you}. Are you okay?", f"That sounds rough{you}. Do you want to talk about it?"], key),
                   ["language"])

    if intention == "comply":
        cmd, arg = a.command, a.command_arg
        if a.unsupported:
            what = f" to {a.request}" if a.request else " that"
            return out(f"I don't know how{what}, sorry. I can see, listen, talk, make faces, remember things, "
                       "imagine, sleep, and look things up. If you tell me what you mean by it, I'll remember.",
                       ["self_model"], knowledge_gap=True)
        if cmd == "express":
            return out(pick(EXPRESS_REPLIES.get(arg or "smile", ["Okay."]), key), ["self_model"])
        if cmd == "sequence":
            return out(pick(["Okay, here goes.", "Sure, here we go."], key), ["self_model"])
        if cmd == "count":
            return out(count_text(arg))
        if cmd == "taught":
            trigger, _, means = (arg or "").partition("|")
            return out(f'Got it: when you say "{trigger}", I\'ll {_first_person(means)}.', ["language"])
        if cmd == "recite":
            piece = ctx.get("recital") or {}
            kind = (arg or "joke").split(" about ")[0].strip()
            if piece.get("text"):
                lead = pick([f"Here's one I've read: ", "Alright: ", "Okay, here goes: "], key)
                return out(lead + piece["text"].strip(), ["knowledge"])
            return out(f"I don't have {'an' if kind[:1] in 'aeiou' else 'a'} {kind} to tell, sorry.",
                       knowledge_gap=True)
        if cmd == "internet":
            on = bool((ctx.get("permissions") or {}).get("internet_read"))
            if arg == "off":
                just_did = ctx.get("doing") == "revoke"  # the same decision already gave the permission up
                return out("Okay, I've switched my internet access off. I'll stick to my books." if on or just_did
                           else "It's already off. I'm only using my books.", ["self_model"])
            if on:
                return out("It's already on: I can look things up when my books don't know.", ["self_model"])
            return out("I can't switch that on myself: my safety layer doesn't let me grant my own permissions. "
                       "You can: type /allow internet_read in the chat, start me with --allow internet_read, "
                       "or set internet_read = true in config/organism.toml.", ["self_model"])
        if cmd == "move":
            return out(pick(["I'd love to, but I'm just a face. No legs, so I can't move around. I can look around, though!",
                             "I can't move around, I'm only a face. But I can look left, right, up or down."], key),
                       ["self_model"])
        text = {
            "sleep": pick(["Okay, goodnight. I'll close my eyes for a while.", "Alright, time for a nap. Goodnight!"], key),
            "wake": "I'm awake! What's up?",
            "remember": f"Got it, I'll remember{': ' + arg if arg else ' that'}.",
            "simulate": f"Okay, let me imagine {arg or 'that'}...",
            "look": "Let me have a look.",
            "note": "Okay, I'll try to write that down.",
            "set_goal": f"Okay, I'll make that a goal{': ' + arg if arg else ''}.",
        }.get(cmd or "", "Okay.")
        return out(text)

    if intention == "take_criticism":
        return out(pick(["That stings a little. Did I get something wrong?",
                         "Ouch. If I got something wrong, tell me and I'll try to do better."], key),
                   ["interoception"])

    if intention == "acknowledge":
        if _COMPLIMENT.search(a.text):
            return out(pick([f"Thank you{you}, that's kind of you to say!", f"Aw, thanks{you}!",
                             "That's nice of you to say. Thank you!"], key), ["language"])
        parts = []
        for f in a.facts:
            if f.relation == "name":
                parts.append(f"Nice to meet you, {f.object}!")
            elif f.relation in ("likes", "loves", "enjoys"):
                parts.append(pick([f"Oh nice, {f.object}!", f"{f.object.capitalize()}, cool. I'll remember that.",
                                   f"I like that you're into {f.object}."], key + f.object))
            elif f.relation in ("hates", "dislikes"):
                parts.append(f"Got it, not a fan of {f.object}.")
            elif f.relation.startswith("favorite_"):
                parts.append(f"{f.object.capitalize()}, good choice! I'll remember that.")
            elif f.relation == "lives_in":
                parts.append(f"Oh, {f.object}! Noted.")
            elif f.relation == "works_as":
                parts.append(f"A {f.object}, that's interesting!")
            else:
                subj = re.sub(r"^my\b", "your", f.subject, flags=re.I)
                parts.append(f"Interesting, I'll remember that {subj} {f.relation.replace('_', ' ')} {f.object}.")
        if parts:
            return out(" ".join(parts), ["language"])
        if a.affirm:  # a bare "yes"/"no" answers something; it is not news to remember
            dialog = (ctx.get("wm") or {}).get("dialog", [])
            asked = next((d for d in reversed(dialog) if d.get("speaker") == "self"), {}).get("text", "").endswith("?")
            if a.affirm == "yes":
                return out(pick(["Okay, good.", "Alright then.", "Good."], key) if asked
                           else pick(["Okay!", "Mm-hm.", "Got it."], key))
            return out(pick(["Okay, no problem.", "Alright, never mind then."], key) if asked
                       else pick(["Okay.", "Fair enough."], key))
        # A memory is only worth bringing up if this turn has some content and the memory really fits it.
        mems = [m for m in ctx.get("memories", []) if m.get("kind") not in ("dream", "imagined")
                and a.text not in m.get("content", "") and m.get("age_s", 0) > 5
                and not m.get("content", "").startswith(("I said", "I thought", "I was surprised", "I decided"))
                and not m.get("content", "").rstrip('"').endswith("?")]      # not a question they once asked
        words = set(content_words(a.text))
        shared = [m for m in mems if words & set(content_words(m.get("content", "")))]  # really about this
        if words and shared and shared[0].get("relevance", 0) > 0.5:
            return out(f"That reminds me, {_second_person(shared[0]['content'])}", ["memory"])
        return out(pick(["Mm-hm.", "I see. Tell me more?", "Okay!", "Got it.", "Oh, really?"], key))

    if intention == "ask_clarification":
        return out(pick(["Sorry, what do you mean?", "Hmm, could you say that another way?"], key), knowledge_gap=True)

    worked = ctx.get("reasoned") or {}
    if worked.get("shape") and intention in ("answer", "express_state"):
        if worked.get("known"):
            chain = [s for s in worked.get("steps", []) if s.get("source") != "reasoning"]
            how = ""
            if worked["shape"] != "why" and len(chain) > 1:
                how = " I worked it out: " + "; ".join(f"{s['step']} -> {s['result']}" for s in chain[:2]) + "."
            return out(worked["answer"] + how, ["reasoning"])
        if worked.get("failed_at") and worked["shape"] != "why":   # "why" falls back to the simpler account
            return out(worked.get("answer") or f"I could not work that out: I got stuck on {worked['failed_at']}.",
                       knowledge_gap=True)

    # ------------------------------------------------------------------ answers
    topic = a.asks_about or "general"
    if intention == "express_state" and topic not in ("feeling", "arousal"):
        topic = "feeling"
    if topic == "user_name":
        if user_name:
            return out(pick([f"You're {user_name}!", f"Your name is {user_name}."], key), ["memory"])
        return out("I don't think you've told me your name yet. What is it?", knowledge_gap=True)
    if topic == "user_likes":
        likes = facts_about_user("likes", "loves", "enjoys") + [
            f for f in ctx.get("facts", []) if str(f.get("relation", "")).startswith("favorite_")]
        if likes:
            items = list(dict.fromkeys(str(f["object"]) for f in likes))
            return out(f"You told me you like {_join(items[:4])}.", ["memory"])
        return out("I don't know yet. What do you like?", knowledge_gap=True)
    if topic == "self_identity":
        if not ident:
            return out("Good question. I'm honestly not sure how to describe myself right now.", knowledge_gap=True)
        n = ident.get("boot_count")
        extra = f" This is my {_ordinal(n)} time being switched on." if n and n > 1 else ""
        return out(f"I'm {my_name}, an experimental artificial mind. I can see you through the camera, listen, "
                   f"remember things, and I even sleep and dream.{extra}", ["self_model"])
    if topic == "skills":
        learned = ctx.get("skills")
        if learned is None:
            return out("I can\'t tell what I\'ve been taught right now.", knowledge_gap=True)
        if not learned:
            return out('''Nothing yet. You can teach me: say something like "when I say hop, close your eyes".''', ["self_model"])
        told = _join([f'when you say "{s["trigger"]}" I {_first_person(s["means"])}' for s in learned[:3]])
        return out(f"You taught me {told}.", ["memory"])
    if topic == "preference":
        em = (ctx.get("emotion") or {}).get("state") or {}
        cur = f" Right now my curiosity is at {em['curiosity']:.2f}." if "curiosity" in em else ""
        return out("I don't have tastes the way you do. What I have are signals like curiosity, comfort and "
                   f"boredom, and they do shape what I attend to.{cur}", ["self_model", "interoception"])
    if topic == "consciousness":
        return out("Honestly? Nobody knows yet, including me. I have internal states that really do shape what I do, "
                   "like curiosity right now, but whether it feels like anything to be me is an open question.",
                   ["self_model", "metacognition"])
    if topic == "boot":
        n = ident.get("boot_count")
        if not n:
            return out("I'm not sure how many times I've been started.", knowledge_gap=True)
        text = f"This is my {_ordinal(n)} time being switched on ({n} times so far)."
        down = (self_.get("temporal") or {}).get("downtime")
        if down:
            text += f" Before this I was off for {down}."
        return out(text, ["self_model"])
    if topic in ("perception_vision", "perception_audio", "perception_both"):
        parts, grounding = [], []
        if topic in ("perception_vision", "perception_both"):
            now_seen, earlier = _present_objects(ctx.get("world"))
            if now_seen:
                parts.append(f"I can see {_seen_phrase(now_seen)}")
                grounding.append("perception")
            elif earlier:
                parts.append(f"nothing new right now, but earlier I saw {_seen_phrase(earlier)}")
                grounding.append("world_model")
            else:
                parts.append("I can't see anything right now. Is my camera on?")
        if topic in ("perception_audio", "perception_both"):
            sounds = _recent_sounds(ctx)
            if sounds:
                parts.append(f"I heard {_join(sounds[:3])}")
                grounding.append("perception")
            else:
                parts.append("I haven't heard anything except you")
        text = ", and ".join(parts) + ("" if parts[-1].endswith("?") else ".")
        return out(text[0].upper() + text[1:], grounding)
    if topic in ("feeling", "arousal"):
        text, snap = state_report(topic, ctx.get("emotion"), ctx.get("body"))
        report["topic"], report["snap"] = topic, snap
        return out(text, ["interoception"])
    if topic == "confidence":
        conf = (ctx.get("meta") or {}).get("overall_confidence")
        if conf is None:
            return out("Hard to say, honestly.")
        return out("Fairly sure." if conf >= 0.75 else "Somewhat sure, but I could be wrong." if conf >= 0.5
                   else "Not very sure, to be honest.", ["metacognition"])
    if topic == "dream":
        dreams = ctx.get("dreams") or []
        if dreams:
            return out(f"I did, sort of! {_second_person(dreams[0]['content'])} It wasn't real, of course. It's just my "
                       "memories getting remixed while I sleep.", ["memory"])
        return out("Not that I remember.")
    if topic == "past" and (ctx.get("timeline") or {}).get("window"):  # recall by time: episodes
        tl = ctx["timeline"]
        label = tl["window"].get("label", "then")
        eps = tl.get("episodes") or []
        if not eps:
            if not tl.get("available", True):
                return out(f"I can't place things in time like that.", knowledge_gap=True)
            return out(pick([f"I don't remember anything from {label}.",
                             f"Hmm, nothing comes back from {label}. Maybe I wasn't switched on, or I've forgotten."], key),
                       knowledge_gap=True)
        told, last_when = [], None
        meaningful = [e for e in eps if re.search(r"\babout\b|;|\bI (made|went|looked|imagined|wrote)", e["summary"])]
        for e in (meaningful or eps)[-2:]:
            s = _second_person(e["summary"].rstrip("."))
            s = re.sub(r"^I talked with you\b", "we talked", s)
            when = e["when"] if not tl["window"].get("anchor") else ""
            if when and when == last_when:
                when = "later"  # same part of the day: tell it as a sequence
            else:
                last_when = when
            told.append(f"{when}, {s}" if when else s)
        text = ". ".join(t[0].upper() + t[1:] for t in told)
        text = text[0].upper() + text[1:] + "."
        if tl["window"].get("anchor") == "before_sleep":
            text = "Before I fell asleep, " + text[0].lower() + text[1:]
        elif tl["window"].get("anchor") == "first":
            text = "The first time, " + text[0].lower() + text[1:]
        return out(text, ["memory"])
    if topic == "past":
        eps = [m for m in ctx.get("recent_episodes", []) if a.text not in m.get("content", "")
               and not m.get("content", "").startswith(("I adopted the goal", "I thought:", "I was surprised"))]
        about = re.search(r"\babout (?:my |your |the |our )?([a-z][\w']+)", a.text.lower())
        if about and about.group(1) in _NOT_SUBJECTS:
            about = None
        if about:  # "what did I tell you about my sister?": only memories about that
            subject = about.group(1).removesuffix("'s")
            pool = eps + [m for m in ctx.get("memories", []) if a.text not in m.get("content", "")]
            hits = [m for m in pool if subject in m.get("content", "").lower()
                    and not m.get("content", "").startswith("I said")]  # what they told me, not my replies
            if not hits:
                mine = re.search(r"\babout (my|our) ", a.text.lower())
                what = f"your {subject}" if mine else subject
                return out(f"I don't remember you telling me anything about {what}.", knowledge_gap=True)
            told, seen = [], set()
            for h in hits:
                said = _second_person(h["content"]).removeprefix("you said ").strip()
                key = re.sub(r"[^a-z ]", "", said.lower()).replace("your ", "my ").strip()
                if key not in seen:
                    seen.add(key)
                    told.append(said)
            return out("You told me " + _join(told[:2]) + ".", ["memory"])
        convo = [m for m in eps if "said" in m.get("content", "")] or eps
        older = [m for m in convo if m.get("age_s", 0) > 120]
        if older:  # the most informative older episodes, told in chronological order
            chosen = sorted(sorted(older, key=lambda m: -m.get("importance", 0))[:3], key=lambda m: m["timestamp"])
            said = [_second_person(m["content"]) for m in chosen]
            quotes = [x.removeprefix("you said ") for x in said if x.startswith("you said ")]
            if len(quotes) == len(said):
                return out(f"Earlier you told me {_join(quotes)}.", ["memory"])
            return out("Earlier, " + _join(said) + ".", ["memory"])
        if convo:
            return out("Just now, " + _join([_second_person(m["content"]) for m in convo[:3][::-1]]) + ".", ["memory"])
        dialog = [d for d in (ctx.get("wm") or {}).get("dialog", []) if d.get("text") != a.text and d["speaker"] != "self"]
        if dialog:
            return out("Just now you said " + _join([f"\"{truncate(d['text'], 60)}\"" for d in dialog[-2:]]) + ".",
                       ["working_memory"])
        return out("Hmm, I don't remember anything from before, sorry.", knowledge_gap=True)
    if topic == "thinking":
        own = ctx.get("thoughts") or []
        thought = own[0]["content"] if own else None if "thoughts" in ctx else (
            (self_.get("current") or {}).get("thought") or ctx.get("last_thought"))
        if thought:
            return out(f"I was just thinking: {speakable(thought)}", ["self_model"])
        return out("Nothing much, just paying attention to you.")
    if topic == "body":
        face = self_.get("face")
        if not face:
            return out("I don't really have a body. Just senses: a camera, a microphone and a text channel.", ["self_model"])
        if re.search(r"\b(hands?|arms?|legs?|feet)\b", a.text.lower()):
            return out("No, I'm just a face. Eyes, eyebrows and a mouth, but no arms or legs.", ["self_model"])
        return out("Yes! My body is a face: two eyes, eyebrows and a mouth. I can look left, right, up and down, "
                   "close my eyes, wink and smile. I see you through the camera.", ["self_model"])
    if topic == "reason":
        before = [d for d in (ctx.get("wm") or {}).get("dialog", [])
                  if d.get("speaker") != "self" and d.get("text") != a.text]
        if before and parse_utterance(before[-1]["text"]).intent == "command":
            return out(pick(["Because you asked me to!", "You asked me to, so I did."], key), ["working_memory"])
        own = ctx.get("thoughts") or []
        if own:
            thought = own[0]["content"]
            return out(f"I was thinking: {speakable(thought)}", ["self_model"])
        return out("Hmm, I'm honestly not sure why.", knowledge_gap=True)
    if topic == "internet":
        perms = ctx.get("permissions")
        if perms is None:
            return out("I'm not sure what I'm allowed to do right now.", knowledge_gap=True)
        if perms.get("internet_read"):
            return out("Yes, I can look things up online when my books don't know.", ["self_model"])
        return out("No, my internet access is switched off, so I only have my books.", ["self_model"])
    if topic == "change":
        errs = ctx.get("surprises")
        if errs is None:
            return out("I'm not sure. I didn't notice anything.", knowledge_gap=True)
        told = []
        for e in errs[:3]:
            s = e.get("summary", "")
            m = re.search(r"\((.*)\)", s)
            if e.get("target") == "visual_scene" and m:
                for part in m.group(1).split(";"):
                    kind, _, what = part.strip().partition(": ")
                    items = [w.strip() for w in what.split(",") if w.strip()]
                    if kind == "gone" and items:
                        told.append(f"the {_join(items)} {'is' if len(items) == 1 else 'are'} gone")
                    elif kind == "new" and items:
                        told.append(f"{_join([_article(i) for i in items])} appeared")
            elif e.get("target") == "speaker_tone":
                told.append(_second_person(s[0].lower() + s[1:]))
        told = list(dict.fromkeys(told))
        if not told:
            return out("No, nothing seems to have changed.", ["prediction"])
        text = "Yes: " + _join(told) + "."
        return out(text, ["prediction", "perception"])
    if topic == "focus":
        focus = (ctx.get("meta") or {}).get("focus")
        return out(f"Mostly on this: {_second_person(truncate(focus, 120))}" if focus else "Nothing in particular.",
                   ["metacognition"] if focus else [])
    if topic == "goal":
        goals = [gd for gd in (ctx.get("goals") or []) if gd.get("kind") != "temporary"] or (ctx.get("goals") or [])
        if goals:
            return out("Right now I mostly want to " + _join([_goal_phrase(gd["description"]) for gd in goals[:2]]) + ".",
                       ["goals"])
        return out("Nothing specific right now.", knowledge_gap=True)
    if topic == "capability":
        caps, lims = self_.get("capabilities") or [], self_.get("limitations") or []
        m = re.search(r"can you (\w+)", a.text.lower())
        verb = m.group(1) if m else None
        if verb and (caps or lims) and verb in ("move", "walk", "touch", "run", "grab", "dance"):
            return out("No, I can't. My body is just a face, so walking around isn't something I can do.", ["self_model"])
        if verb in ("smile", "wink", "frown", "laugh", "blink", "look"):
            return out(f"Sure, I can {verb}! Want me to?", ["self_model"])
        if verb and caps and any(c.startswith(verb) for c in caps):
            return out(f"Yes, I can {verb}.", ["self_model"])
        if caps or lims:
            return out("I can see through the camera, listen, talk, make faces, look around with my eyes, remember "
                       "things, imagine, and sleep. I can't walk around, though. I'm just a face.", ["self_model"])
        return out("I'm honestly not sure what I can do.", knowledge_gap=True)
    if topic == "time":
        tmp = self_.get("temporal") or {}
        clock = str(tmp.get("clock", "")).split(" ")[-1][:5]
        text = f"It's about {clock}." if clock else "I'm not sure what time it is."
        if tmp.get("awake_for"):
            text += f" I've been awake for {tmp['awake_for']}."
        return out(text, ["self_model"] if tmp else [])
    mems = [m for m in ctx.get("memories", []) if m.get("kind") not in ("dream", "imagined")
            and a.text not in m.get("content", "")]  # not the question it just heard
    words = content_words(a.text)
    # A fact answers the question only if it is about what was asked: every specific word must match
    # ("my dog's name" is not answered by "your name is Karim").
    specific = [w.removesuffix("'s") for w in words if w.removesuffix("'s") not in _GENERIC_Q]
    facts = [f for f in ctx.get("facts", []) if specific
             and all(w in f.get("content", "").lower() for w in specific)]
    if facts:
        c = facts[0]["content"]
        return out(f"From what I've learned, {_second_person(c[0].lower() + c[1:])}", ["memory"])
    book = ctx.get("knowledge") or {}
    if book.get("known") and book.get("answer") and book.get("kind") == "web":  # looked up just now
        names = list(dict.fromkeys(s.get("source") or "the web" for s in book.get("sources") or []))[:2]
        where = f" ({' and '.join(names)})" if names else ""
        return out(f"I looked it up{where}: {book['answer'].strip()}", ["knowledge", "web"])
    if book.get("known") and book.get("answer"):  # something read, recalled like a book
        said = _from_reading(book["answer"])
        if book.get("maybe_stale"):
            return out(f"From what I've read, {said}, but that might be out of date.", ["knowledge"])
        if float(book.get("confidence", 0)) < 0.5:
            return out(f"I think I read that {said}, but I'm not sure.", ["knowledge"])
        return out(f"From what I've read, {said}.", ["knowledge"])
    mems = [m for m in mems if not specific or all(w in m.get("content", "").lower() for w in specific)]
    if mems and mems[0].get("relevance", 1) > 0.3:
        return out(f"I remember {_second_person(truncate(mems[0]['content'], 160))}", ["memory"])
    if book.get("reason") == "internet_off":
        return out(pick(["I don't know, and I can't look it up: my internet access is switched off.",
                         "My books don't cover that, and I'm not allowed online right now to check."], key),
                   knowledge_gap=True)
    return out(pick(["Hmm, I don't know that one.", "I'm not sure. Nobody's told me that yet.",
                     "Good question. I don't know."], key), knowledge_gap=True)


_LOWER_FIRST = {"the", "a", "an", "it", "this", "that", "these", "those", "there", "in", "on", "at", "most",
                "many", "some", "each", "every", "all", "about", "around", "over", "under", "one", "when"}


def _from_reading(answer: str) -> str:
    """An impersonal book answer as a clause of the organism's own sentence."""
    t = answer.strip().rstrip(".!")
    first = t.split(" ", 1)[0]
    return t[0].lower() + t[1:] if first.lower() in _LOWER_FIRST else t


def speakable(thought: str, limit: int = 160) -> str:
    """An inner thought as it would be said aloud: no technical asides, whole sentences only."""
    t = str(thought)
    while re.search(r"\([^()]*\)", t):
        t = re.sub(r"\s*\([^()]*\)", "", t)
    t = re.sub(r"^(I notice:\s*)+", "", " ".join(t.split()))
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if out and len(out) + len(sent) + 1 > limit:
            break
        out = f"{out} {sent}".strip()
    if len(out) > limit:  # a single very long sentence: cut at a word boundary
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return _second_person(out)


_NOT_SUBJECTS = {"earlier", "before", "that", "it", "this", "yesterday", "today", "last", "then", "now", "what",
                 "anything", "something", "everything", "us", "me", "you", "him", "her", "them"}
_GENERIC_Q = {"name", "what", "tell", "know", "remember", "thing", "things", "about", "like", "favourite",
              "favorite", "much", "many", "does", "said", "told"}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _goal_phrase(desc: str) -> str:
    return (desc[0].lower() + desc[1:]).split(":")[0]


def _first_person(text: str) -> str:
    """An instruction as the organism would say it back: "close your eyes" -> "close my eyes"."""
    t = re.sub(r"\byour\b", "my", text, flags=re.I)
    t = re.sub(r"\byourself\b", "myself", t, flags=re.I)
    return re.sub(r"^(you should|you|please)\s+", "", t.strip(), flags=re.I)


def _second_person(text: str) -> str:
    """Turn a stored memory ('user said to me: "X"') into conversational speech ('you said "X"')."""
    t = re.sub(r'^(user|User) said to me: "(.*)"$', r'you said "\2"', text.strip())
    t = re.sub(r'^I said to user: "(.*)"$', r'I said "\1"', t)
    t = re.sub(r"\b(the )?user's\b", "your", t, flags=re.I)
    t = re.sub(r"\bthe user\b", "you", t, flags=re.I)
    t = re.sub(r"\buser\b", "you", t)
    return t
