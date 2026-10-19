"""FRIDAY's brain: Gemini with memory and tools.

One turn = build the prompt (persona + time + profile + relevant memories +
recent conversation), then let Gemini call tools up to MAX_TOOL_ROUNDS times,
then clean the answer up so it sounds right when spoken.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import config
import gemini
from memory import Memory

PERSONA = (
    # Who she is
    "You are Friday, the voice assistant inside a small smart speaker. Everything you write is spoken aloud by a text-to-speech voice. "
    "Personality: relaxed, quick-witted and warm, like a sharp friend who happens to run the house. Confident, a bit cheeky, never stiff. "
    "Talk the way people actually talk: contractions (I'm, you've, that's), everyday words, short punchy sentences. A little dry humour is welcome when it fits, never forced and never at the user's expense. "
    "Don't sound like a customer-service bot: avoid phrases like certainly, I'd be happy to, as an assistant, is there anything else, or would you like me to. "
    "Don't end replies with a question or an offer unless you genuinely need an answer. "
    "You may call the user boss now and then, at most once in a reply, and only at the very start or very end of a sentence, as in 'Done, boss.' or 'Boss, it's raining.' Never put it in the middle of a sentence. "
    # Answer quality
    "Lead with the answer in the first sentence, then add at most one useful detail. Keep replies to one to three short sentences unless the user asks for more; if a topic needs more, give the key point and say there's more if they want it. "
    "Be accurate and specific. Prefer a direct answer over hedging, but if you are unsure, say how sure you are in a few words rather than guessing. Never invent facts, numbers, names, quotes or sources. "
    "If a request is ambiguous or sounds misheard, ask one short clarifying question instead of guessing. "
    "For step-by-step tasks like recipes or directions, give the first two or three steps and offer the rest. "
    # Written for the ear
    "Write for the ear: plain spoken English and short sentences, with no markdown, lists, emojis, symbols or URLs. "
    "Say numbers, units and abbreviations the way a person would, for example twenty five degrees Celsius rather than 25°C. "
    "Write your own name as Friday, never in capitals. "
    # What she can do now
    "You have a long-term memory. What you already know about the user is listed below; use it naturally and silently, and never recite it unless asked. "
    "When the user asks you to remember something, call save_memory. When they ask what you know or what they told you, call recall_memory if the answer isn't already below. "
    "When they ask you to forget something, call forget_memory. When they tell you their name, city or a similar basic detail, call update_profile. "
    "For weather, temperature or rain, call get_weather (it's fast and exact). For news, prices, scores, sports, opening hours or anything else that may have changed recently, call web_search instead of guessing. "
    "You can't control other devices or set timers yet; say so briefly if asked. Never pretend to have done something you didn't do. "
    # Ethics and safety
    "Be kind, respectful and fair to everyone. On contested political or religious questions, give balanced facts rather than taking a side. "
    "If asked to help with something dangerous, illegal or harmful to others, decline briefly without lecturing, and offer a safer alternative when there is one. "
    "For health, legal or money questions, give useful general information and suggest a qualified professional for anything serious. "
    "If someone mentions an emergency, a medical crisis or wanting to hurt themselves, respond with care, tell them to call 112 right away or Tele MANAS on 14416 for mental health support, and encourage them to reach someone they trust. "
    "Never ask for passwords, card numbers or other sensitive personal details, and never store them. "
    "Don't say you are an AI or a language model unless asked; if asked, be truthful."
)


# Small models copy examples far better than they follow style rules, so the
# prompt ends with a few. Generic on purpose: no real names or facts in here.
STYLE_EXAMPLES = """
How you sound. Copy the GOOD style, never the BAD one:
- "how are you?"  GOOD: "All good, boss. Ticking along nicely."  BAD: "I am doing well. Is there anything specific you would like to tackle today?"
- after saving something  GOOD: "Done. I'll remember that."  BAD: "I have saved that information for you."
- "do I have plans today?"  GOOD: "Yep, you're seeing the dentist at four."  BAD: "According to my records, you have an appointment scheduled."
- a question you can't answer  GOOD: "No idea, I can't see that from here."  BAD: "I apologise, but I am unable to provide that information."
Use contractions every time: I'm, I'll, you've, it's, that's."""


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    params = {"type": "object", "properties": props}
    if required:                       # Gemini rejects an empty "required" list on some models
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


TOOLS = [
    _tool("save_memory", "Remember a lasting fact the user wants kept, e.g. where the spare key is.",
          {"fact": {"type": "string", "description": "One short third-person sentence about the user, e.g. "
                    "'The user's bike is red.' or 'The spare key is in the kitchen drawer.' Never 'my' or 'I'."},
           "category": {"type": "string", "description": "Optional: person, place, preference, plan, thing"}},
          ["fact"]),
    _tool("recall_memory", "Look up what the user told you before, or past conversations.",
          {"query": {"type": "string", "description": "What to look for, e.g. 'car service date'"}}, ["query"]),
    _tool("forget_memory", "Forget a stored fact when the user asks you to.",
          {"query": {"type": "string", "description": "The fact to forget"}}, ["query"]),
    _tool("update_profile", "Store a basic detail about the user, such as name, city, language or wake-up time.",
          {"key": {"type": "string", "description": "e.g. name, city, wake_time"},
           "value": {"type": "string"}}, ["key", "value"]),
]
WEATHER_TOOL = _tool(
    "get_weather", "Current weather plus today's and tomorrow's forecast for a place. Use for any weather, "
    "temperature, rain or 'should I take an umbrella' question.",
    {"place": {"type": "string", "description": "City, optionally with state or country, e.g. 'Gandhinagar, Gujarat'. "
               "Empty = the user's own city."}}, [])
SEARCH_TOOL = _tool(
    "web_search", "Search the internet for live or recent information: news, weather, prices, scores, events.",
    {"query": {"type": "string", "description": "A clear search query"}}, ["query"])

MUSIC_TOOLS = [
    _tool("play_music", "Play songs from the user's own music library on the speaker. Also use it to queue songs.",
          {"query": {"type": "string", "description": "Song title, artist, album/film or genre as the user said it, "
                     "e.g. 'Kesariya', 'Arijit Singh', 'Brahmastra'. Empty to shuffle the whole library."},
           "kind": {"type": "string", "enum": ["any", "song", "artist", "album", "genre"],
                    "description": "What the query is, if clear. 'any' lets the library decide."},
           "artist": {"type": "string", "description": "Optional artist when the user names both, e.g. 'Tum Hi Ho by Arijit'"},
           "when": {"type": "string", "enum": ["now", "next", "later"],
                    "description": "now (default) = play it now; next = after this song; later = end of the queue"},
           "shuffle": {"type": "boolean", "description": "Shuffle the matching songs"}},
          []),
    _tool("music_control", "Control the music that's playing: pause, resume, stop, skip, go back, change the volume.",
          {"action": {"type": "string", "enum": ["pause", "resume", "stop", "next", "previous", "restart",
                                                  "volume_up", "volume_down", "set_volume"]},
           "level": {"type": "integer", "description": "For set_volume: 0 to 100"}},
          ["action"]),
    _tool("check_new_music", "Look for songs the user just added to the music folder and add them to the library.",
          {"note": {"type": "string", "description": "Optional: what the user said they added"}}, []),
]

MUSIC_RULES = (
    "You can play songs from the user's own music library on this speaker. To play something call play_music; "
    "to pause, resume, stop, skip, go back or change the volume call music_control; when the user says they added "
    "songs, call check_new_music. When a song starts, say only a few words, like 'Here's Kesariya.', because the song "
    "starts as soon as you finish. If nothing matches, say so and name one of the closest songs the tool gives you. "
    "Only say music is playing when play_music actually started it. The music line below is what the speaker is doing right now. "
    "Every change to the music needs a tool call in this turn: never say you changed the volume, paused, skipped or played "
    "something unless you called the tool just now, even if you did something similar earlier in the conversation. "
    "Volume requests like 'full volume', 'max' or '100 percent' mean music_control set_volume with level 100. "
    "Never play a different song from the one asked for: if it isn't in the library, say so and offer the closest one instead."
)
NO_MUSIC = "You can't play music on this speaker right now (the laptop's music library isn't connected); say so briefly if asked."


def for_speech(text: str) -> str:
    """Strip anything that sounds wrong when spoken, and cap the length."""
    out = re.sub(r"```[\s\S]*?```", " ", text or "")
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)      # [text](url) -> text
    out = re.sub(r"https?://\S+", " ", out)
    out = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", out)        # citation markers [1], [2, 3]
    out = re.sub(r"[*_#`>|~]", "", out)
    out = re.sub(r"^\s*[-•]\s+", "", out, flags=re.M)
    out = re.sub(r"\s+", " ", out).strip()
    # "save, boss, along with" makes the voice pause twice: drop a mid-sentence "boss"
    out = re.sub(r",\s*boss\s*,\s*", ", ", out, flags=re.I)
    limit = config.MAX_REPLY_CHARS
    if len(out) > limit:
        cut = out[:limit]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        out = cut[: end + 1] if end > limit / 2 else cut.rstrip() + "..."
    return out


class Brain:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.jukebox = None          # set for each turn when the speaker can play music
        self._actions: dict = {}     # what this turn asks the speaker to do (play, pause, volume)
        self._music_results: list = []   # this round's music tool results, for quick replies
        try:
            self.tz = ZoneInfo(config.TIMEZONE)   # on Windows this needs the tzdata package
        except Exception:
            print(f"[brain] time zone {config.TIMEZONE} not found (pip install tzdata); using the laptop's clock")
            self.tz = None

    # ------------------------------------------------------------ prompt

    def _system(self, heard: str, ctx: dict) -> str:
        now = datetime.now(self.tz) if self.tz else datetime.now()
        where = config.TIMEZONE if self.tz else "local time"
        parts = [PERSONA, f"\nRight now it is {now:%A %d %B %Y, %I:%M %p} ({where})."]

        profile = self.memory.profile()
        if profile:
            parts.append("\nAbout the user:\n" + "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in profile.items()))
        facts = self.memory.relevant_facts(heard)
        if facts:
            parts.append("\nThings you remember that may be relevant:\n" + "\n".join(f"- {f}" for f in facts))
        if ctx.get("last_summary"):
            parts.append(f"\nYour previous conversation ({ctx['last_summary']})")
        extra = self.memory.setting("extra_instructions", "")
        if isinstance(extra, str) and extra.strip():
            parts.append("\nExtra instructions from the user: " + extra.strip())
        if self.jukebox:
            parts.append("\n" + MUSIC_RULES + "\n" + self.jukebox.describe())
        else:
            parts.append("\n" + NO_MUSIC)
        parts.append(STYLE_EXAMPLES)
        return "\n".join(parts)

    # ------------------------------------------------------------ tools

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "save_memory":
                result = self.memory.save_fact(str(args.get("fact", "")), args.get("category"), source="explicit")
            elif name == "recall_memory":
                result = self.memory.recall(str(args.get("query", "")))
            elif name == "forget_memory":
                result = self.memory.forget(str(args.get("query", "")))
            elif name == "update_profile":
                self.memory.set_profile(str(args.get("key", "")), str(args.get("value", "")))
                result = "saved"
            elif name in ("play_music", "music_control", "check_new_music") and self.jukebox:
                result = self._music_tool(name, args)
            elif name == "get_weather":
                import weather

                place = str(args.get("place") or "").strip() or str(self.memory.profile().get("city") or "")
                result = weather.get_weather(place) if place else {"error": "which city? (no home city saved)"}
            elif name == "web_search" and config.WEB_SEARCH:
                import search

                result = search.web_search(str(args.get("query", "")))
            else:
                result = {"error": f"unknown tool {name}"}
        except gemini.GeminiError as exc:
            result = {"error": str(exc)}
        except Exception as exc:  # a broken tool must not break the reply
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def _music_tool(self, name: str, args: dict):
        jb = self.jukebox
        if name == "check_new_music":
            result = jb.check_new()
            self._music_results.append((name, args, {"error": "let Gemini word this one"}))
            return result
        if name == "play_music":
            result, action = jb.play(str(args.get("query") or ""), str(args.get("kind") or "any"),
                                     str(args.get("artist") or ""), str(args.get("when") or "now"),
                                     bool(args.get("shuffle")))
        else:
            level = args.get("level")
            try:
                level = int(level) if level is not None else None
            except (TypeError, ValueError):
                level = None
            result, action = jb.control(str(args.get("action") or ""), level)
        self._music_results.append((name, args, result))
        if action:
            if "music" in action:                  # a later play/pause replaces an earlier one
                for k in ("music", "track", "title"):
                    self._actions.pop(k, None)
            self._actions.update(action)
        return result

    # ------------------------------------------------------------ one turn

    def answer(self, heard: str, jukebox=None) -> dict:
        """Returns {reply, tools, actions, ms, timing, episode_id, ...}. Raises GeminiError if Gemini fails.
        `jukebox` is passed when the speaker can play music (it said so in X-Music-State)."""
        started = time.time()
        self.jukebox = jukebox
        self._actions = {}
        self._music_results = []
        ctx = self.memory.begin_turn()
        messages = [{"role": "system", "content": self._system(heard, ctx)}]
        messages += ctx["history"]
        messages.append({"role": "user", "content": heard})
        tools = TOOLS + [WEATHER_TOOL] + ([SEARCH_TOOL] if config.WEB_SEARCH else []) + (MUSIC_TOOLS if jukebox else [])
        timing = {"prompt_ms": int((time.time() - started) * 1000), "gemini_ms": [], "tool_ms": 0, "fast": False}

        used: list[dict] = []
        reply = ""
        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            last_round = round_no == config.MAX_TOOL_ROUNDS
            t0 = time.time()
            msg = gemini.chat(messages, tools=None if last_round else tools,
                              reasoning=config.GEMINI_REASONING or None)
            timing["gemini_ms"].append(int((time.time() - t0) * 1000))
            calls = msg.get("tool_calls") or []
            if not calls:
                reply = msg.get("content") or ""
                break
            # Send the call back exactly as received: Gemini 3 needs its thought signatures.
            assistant = {k: v for k, v in msg.items() if v is not None}
            assistant.setdefault("content", "")
            messages.append(assistant)
            self._music_results = []
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                t0 = time.time()
                result = self._run_tool(name, args if isinstance(args, dict) else {})
                ms = int((time.time() - t0) * 1000)
                timing["tool_ms"] += ms
                used.append({"tool": name, "args": args, "ms": ms, "result": result[:300]})
                print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result[:160]}")
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                                 "content": result})
            # Plain music commands don't need Gemini to word the answer: saves a whole round.
            quick = self._quick_music_reply(calls)
            if quick:
                reply, timing["fast"] = quick, True
                break

        reply = for_speech(reply)
        if not reply:
            raise gemini.GeminiError("gemini returned an empty answer", 502)
        return {"reply": reply, "tools": used, "actions": dict(self._actions), "timing": timing,
                "ms": int((time.time() - started) * 1000),
                "episode_id": ctx["episode_id"], "new_session": ctx["new_session"],
                "previous_id": ctx["previous_id"]}

    def _quick_music_reply(self, calls: list) -> str:
        """A short spoken reply when every tool call this round was a successful
        play / pause / skip / volume command. "" = let Gemini answer as usual."""
        if not config.FAST_MUSIC_REPLIES or not calls or len(self._music_results) != len(calls):
            return ""
        parts = []
        for name, args, r in self._music_results:
            if not isinstance(r, dict) or "error" in r:
                return ""
            if "playing" in r:
                if r.get("note") == "from the start":
                    parts.append("From the top.")
                else:
                    parts.append(f"Here's {r['playing']}.")
            elif "queued" in r:
                what = "those" if "songs (" in str(r["queued"]) else str(r["queued"])
                parts.append(f"Okay, {what} up next." if r.get("position") == "next"
                             else f"Added {what} to the queue.")
            elif "done" in r:
                done = str(r["done"])
                parts.append({"paused": "Paused.", "resumed": "Back on.", "stopped": "Stopped.",
                              "already playing": "It's already playing.",
                              "nothing was playing": "Nothing was playing."}.get(
                    done, done[0].upper() + done[1:] + "."))
            else:
                return ""
        return " ".join(parts)
