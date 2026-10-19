"""FRIDAY local server: the laptop does the hearing, the voice and the memory;
Gemini does the thinking.

It speaks exactly the same HTTP protocol as the Cloudflare Worker (/ask, /say,
/chat, /health), so the S3 can use either one. The S3 finds this laptop by UDP
broadcast and falls back to the Worker when the laptop is off. Music is laptop
only: GET /music/<id> and /music/next stream songs from your library.

    python server.py                       run it
    python server.py --test "hi there"     check Gemini, memory and the voice, no hardware
    python server.py --test-wav q.wav      the whole pipeline on a recording
    python server.py --test "play kesariya" --music   as if the speaker can play music
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

import config
import frames
import music
from audio_utils import bytes_to_pcm, load_wav, normalize, resample, save_wav, split_sentences
from replystream import StreamingReply

HERE = Path(__file__).resolve().parent
NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?"
# Said while a slow tool runs (web search, weather), and when a streamed reply breaks off half way
FILLERS = ["One sec, boss.", "Let me check.", "Checking that now.", "Just a moment.", "On it, boss."]
ERROR_LINE = "Sorry, something went wrong, boss."
QUICK_LINES = (["Paused.", "Back on.", "Stopped.", "It's already playing.", "Nothing was playing.", "From the top.",
                NOT_HEARD_REPLY, ERROR_LINE] + FILLERS + [f"Volume {n} percent." for n in range(0, 101, 5)])
MAX_BODY = config.MAX_AUDIO_SECONDS * config.SAMPLE_RATE * 2 + 64 * 1024
CHUNK = 16 * 1024


class TurnError(Exception):
    """A turn that failed; becomes a JSON error reply the S3 can print."""

    def __init__(self, message: str, status: int = 502, **extra) -> None:
        super().__init__(message)
        self.status = status
        self.extra = extra


# ====================================================================
#  The assistant: hearing -> thinking -> voice
# ====================================================================

class Assistant:
    def __init__(self, with_stt: bool = True) -> None:
        from brain import Brain
        from memory import Background, Memory
        from tts import TTS

        self.turn_lock = threading.Lock()      # one turn at a time: there is one speaker
        self.stream_fails = 0                  # streamed turns in a row that broke before sending anything
        self.memory = Memory()
        self.brain = Brain(self.memory)
        self.background = Background()
        self.tts = TTS()
        self.tts.warm_cache(QUICK_LINES)        # "Paused.", "Volume 50 percent."... ready before they're needed
        self.jukebox = None
        try:
            from jukebox import Jukebox

            self.jukebox = Jukebox(self.memory)
        except Exception as exc:              # music is optional: never stop Friday starting
            print(f"[music] off: {type(exc).__name__}: {exc}")
        self.stt = None
        if with_stt:
            from stt import STT

            self.stt = STT()
            if config.STT_WARMUP:
                self.stt.warmup()

    # ------------------------------------------------------------ pieces

    def hear(self, pcm: np.ndarray) -> tuple[str, str, int]:
        """Returns (cleaned transcript, raw transcript, ms)."""
        from stt import clean_transcript

        t0 = time.time()
        raw = self.stt.transcribe(normalize(pcm))
        return clean_transcript(raw), raw, int((time.time() - t0) * 1000)

    def think(self, heard: str, music_state: str | None = None) -> dict:
        """music_state: the speaker's X-Music-State header (None = it can't play music)."""
        import gemini

        jb = self.jukebox if self.jukebox and self.jukebox.observe(music_state) else None
        try:
            return self.brain.answer(heard, jb)
        except gemini.GeminiError as exc:
            raise TurnError(str(exc), exc.status, heard=heard) from None

    def stream_enabled(self) -> bool:
        return config.STREAM_REPLIES and self.stream_fails < config.STREAM_MAX_FAILS

    def think_stream(self, heard: str, music_state: str | None, sink) -> dict:
        """think(), but each sentence goes to `sink` while Gemini is still writing the rest."""
        import gemini

        jb = self.jukebox if self.jukebox and self.jukebox.observe(music_state) else None
        try:
            return self.brain.answer_stream(heard, jb, sink)
        except gemini.GeminiError as exc:
            raise TurnError(str(exc), exc.status, heard=heard) from None

    def voice_start(self, reply: str) -> tuple:
        """(backend, first audio bytes, rest iterator, ms); raises TTSError."""
        return self.tts.start(reply)

    def after_turn(self, heard: str, result: dict) -> None:
        """Log the exchange now (fast), learn from it later (slow, background)."""
        if not heard or not result.get("episode_id"):
            return
        try:
            self.memory.log_turn(result["episode_id"], heard, result["reply"], result.get("tools", []),
                                 result.get("ms", 0))
        except Exception as exc:
            print(f"[memory] couldn't log the turn: {exc}")
        explicit = any(t["tool"] in ("save_memory", "forget_memory", "update_profile")
                       for t in result.get("tools", []))
        if not explicit:
            self.background.submit(self.memory.learn_from_turn, heard, result["reply"])
        if result.get("new_session") and result.get("previous_id"):
            self.background.submit(self.memory.summarise_episode, result["previous_id"])


# ====================================================================
#  HTTP
# ====================================================================

def read_audio(body: bytes) -> np.ndarray:
    """Raw 16 kHz s16le mono from the S3, or a 16-bit WAV file (laptop tests)."""
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        import io
        import wave

        with wave.open(io.BytesIO(body), "rb") as w:
            if w.getsampwidth() != 2:
                raise TurnError("WAV must be 16-bit", 400)
            ch, rate = w.getnchannels(), w.getframerate()
            pcm = bytes_to_pcm(w.readframes(w.getnframes()))
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
        pcm = resample(pcm, rate, config.SAMPLE_RATE)
    else:
        pcm = bytes_to_pcm(body)
    seconds = pcm.size / config.SAMPLE_RATE
    if seconds < config.MIN_AUDIO_SECONDS:
        raise TurnError("audio too short", 400)
    if seconds > config.MAX_AUDIO_SECONDS:
        raise TurnError(f"audio longer than {config.MAX_AUDIO_SECONDS} s", 400)
    return pcm


class _HttpTransport:
    """How a StreamingReply writes its response: a chunked 200, started when the first audio is ready."""

    def __init__(self, handler: "Handler") -> None:
        self.h = handler

    def begin(self) -> None:
        h = self.h
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()

    def write(self, data: bytes) -> None:
        self.h._chunk(data)

    def end(self) -> None:
        self.h.wfile.write(b"0\r\n\r\n")
        self.h.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive: the S3 reuses one connection
    server_version = "friday-local/1.0"
    assistant: Assistant = None            # set in main()

    def log_message(self, fmt, *args) -> None:   # quiet; turns print their own summary
        pass

    # ------------------------------------------------------------ replies

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes) -> None:
        for i in range(0, len(data), CHUNK):
            part = data[i:i + CHUNK]
            self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
        self.wfile.flush()

    def _stream(self, header: bytes, first: bytes, rest, backend) -> bytes:
        """Send the frame header with the first audio straight away, then the
        rest of the reply (same voice) as soon as each piece is ready."""
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent = [first]
        self._chunk(header + first)
        self.first_audio_ms = int((time.time() - getattr(self, "t_start", time.time())) * 1000)
        while True:
            try:
                data = next(rest)
            except StopIteration:
                break
            except Exception as exc:
                print(f"[tts] stopped mid-reply: {exc}")
                break
            self._chunk(data)
            sent.append(data)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return b"".join(sent)

    # ------------------------------------------------------------ routes

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/music/"):
            try:
                return self._music(url)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return                         # the speaker skipped or paused: normal
        if url.path != "/health":
            return self._json({"error": "not found"}, 404)
        a = self.assistant
        jb = a.jukebox
        self._json({"ok": True, "server": "friday-local", "model": config.GEMINI_MODEL,
                    "voice": a.tts.name, "stt": config.WHISPER_MODEL if a.stt else None,
                    "facts": a.memory.count_facts(), "web_search": config.WEB_SEARCH, "stream": a.stream_enabled(),
                    "music": {"songs": jb.count(), "ffmpeg": bool(jb.ffmpeg)} if jb else None})

    # ------------------------------------------------------------ music

    def _music(self, url) -> None:
        """GET /music/<id>?t=<seconds> or /music/next?after=<id>: one song, 16 kHz mono MP3."""
        if not config.DEVICE_TOKEN or self.headers.get("X-Device-Token", "") != config.DEVICE_TOKEN:
            return self._json({"error": "bad or missing X-Device-Token"}, 401)
        jb = self.assistant.jukebox
        if not jb:
            return self._json({"error": "music is off on the laptop (see the [music] line at start-up)"}, 503)
        q = parse_qs(url.query)
        name = url.path[len("/music/"):].strip("/")
        try:
            if name == "next":
                after = int((q.get("after") or ["0"])[0] or 0) or None
                track_id, start = jb.next_after(after), 0.0
                if track_id is None:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    print("[music] queue finished")
                    return
            else:
                track_id = int(name)
                start = max(0.0, float((q.get("t") or ["0"])[0] or 0))
        except ValueError:
            return self._json({"error": "bad track"}, 400)

        t = jb.track(track_id)
        src = jb.source_file(t) if t else None
        if not src:
            return self._json({"error": f"song {track_id} is not in the library (or its file is gone)"}, 404)
        if start < 1:
            jb.played(track_id)
        print(f"[music] {'playing' if start < 1 else f'resuming at {int(start)} s:'} {jb.label(t)}")

        head = {"X-Track-Id": str(track_id), "X-Track-Title": quote(jb.label(t)[:120]),
                "X-Track-Duration": str(t.get("duration_s") or 0)}
        if jb.ffmpeg:
            cache = jb.cache_file(t)
            if cache.exists():
                size = cache.stat().st_size
                offset = min(size, int(start * _bitrate_bytes()))
                return self._send_file(cache, offset, size, head)
            return self._send_ffmpeg(src, start, cache, head)
        span = music.mp3_span(src)
        if not span:
            return self._json({"error": f"{src.name} isn't an MP3; install ffmpeg to play it"}, 503)
        first, end = span
        if start > 0 and t.get("duration_s"):
            first = min(end, first + int((end - first) * start / t["duration_s"]))
        self._send_file(src, first, end, head)

    def _audio_headers(self, head: dict, length: int | None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        for k, v in head.items():
            self.send_header(k, v)
        if length is None:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_file(self, path: Path, first: int, end: int, head: dict) -> None:
        """Bytes [first, end) of a file. The speaker reads at its own pace."""
        self._audio_headers(head, end - first)
        with open(path, "rb") as f:
            f.seek(first)
            left = end - first
            while left > 0:
                data = f.read(min(CHUNK, left))
                if not data:
                    break
                self.wfile.write(data)
                left -= len(data)
        self.wfile.flush()

    def _send_ffmpeg(self, src: Path, start: float, cache: Path, head: dict) -> None:
        """Convert while sending (first play of a song). From the start, the result is
        also saved to the cache, so the next play skips ffmpeg."""
        proc = subprocess.Popen(music.ffmpeg_cmd(src, start), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tee, tmp = None, None
        try:
            first = proc.stdout.read1(CHUNK) if hasattr(proc.stdout, "read1") else proc.stdout.read(CHUNK)
            if not first:
                err = proc.stderr.read().decode(errors="replace").strip()[:200]
                proc.wait()
                print(f"[music] ffmpeg couldn't convert {src.name}: {err}")
                return self._json({"error": f"ffmpeg failed: {err}"}, 500)
            if start < 1:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
                tee = open(tmp, "wb")
            self._audio_headers(head, None)
            data = first
            while data:
                if tee:
                    tee.write(data)
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                data = proc.stdout.read1(CHUNK) if hasattr(proc.stdout, "read1") else proc.stdout.read(CHUNK)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            ok = proc.wait() == 0
            if tee:
                tee.close()
                tee = None
                if ok:
                    try:
                        os.replace(tmp, cache)
                        tmp = None
                    except OSError:            # Windows: another copy is being written or read
                        pass
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if tee:
                tee.close()
            if tmp:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            return self._json({"error": "request too large"}, 413)
        body = self.rfile.read(length) if length else b""
        self.t_start = time.time()             # the question is in: the speaker is now waiting

        if url.path not in ("/ask", "/say", "/chat"):
            return self._json({"error": "not found"}, 404)
        token = self.headers.get("X-Device-Token", "")
        if not config.DEVICE_TOKEN or token != config.DEVICE_TOKEN:
            return self._json({"error": "bad or missing X-Device-Token"}, 401)

        self.music_state = self.headers.get("X-Music-State")   # sent by speakers that can play music
        with self.assistant.turn_lock:
            try:
                if url.path == "/ask":
                    self._ask(body, parse_qs(url.query).get("debug") == ["1"])
                else:
                    try:
                        text = str(json.loads(body or b"{}").get("text", "")).strip()
                    except ValueError:
                        raise TurnError('body must be JSON like {"text": "..."}', 400) from None
                    if not text:
                        raise TurnError("text is empty", 400)
                    if len(text) > config.MAX_INPUT_CHARS:
                        raise TurnError(f"text longer than {config.MAX_INPUT_CHARS} characters", 400)
                    (self._chat if url.path == "/chat" else self._say)(text)
            except TurnError as exc:
                print(f"[turn] failed: {exc}")
                self._json({"error": str(exc), **exc.extra}, exc.status)
            except (BrokenPipeError, ConnectionResetError):
                print("[turn] the speaker hung up mid-reply")
            except Exception as exc:
                print(f"[turn] crashed: {type(exc).__name__}: {exc}")
                try:
                    self._json({"error": f"server error: {exc}"}, 500)
                except OSError:
                    pass

    def _first_voice(self, reply: str, **extra) -> tuple:
        try:
            return self.assistant.voice_start(reply)
        except Exception as exc:
            raise TurnError(f"voice: {exc}", 502, reply=reply, **extra) from None

    def _frame(self, heard: str | None, text: str, chat_ms: int, actions: dict | None, backend, tts_ms: int,
               stt_ms: int) -> bytes:
        """I2S4 (with the music action) for speakers that can play music, else the old frames."""
        if self.music_state is not None:
            return frames.reply_header(heard, text, backend.provider, chat_ms, tts_ms, stt_ms, backend.fmt, actions)
        if heard is None:
            return frames.say_header(text, backend.provider, chat_ms, tts_ms, backend.fmt)
        return frames.ask_header(heard, text, backend.provider, chat_ms, tts_ms, stt_ms, backend.fmt)

    def _header(self, heard: str | None, result: dict, backend, tts_ms: int, stt_ms: int) -> bytes:
        return self._frame(heard, result["reply"], result["ms"], result.get("actions"), backend, tts_ms, stt_ms)

    # ------------------------------------------------------------ a reply that starts while Gemini is still writing

    def _wants_stream(self) -> bool:
        """An X-Stream: 0 or 1 header on the request decides (that's how the two ways are compared);
        otherwise STREAM_REPLIES, unless streaming has broken too often in a row and switched itself off."""
        flag = (self.headers.get("X-Stream") or "").strip().lower()
        if flag:
            return flag not in ("0", "false", "no", "off")
        return self.assistant.stream_enabled()

    def _turn_streamed(self, heard: str, stt_ms: int, typed: bool) -> bool:
        """Answer `heard` with the audio starting while Gemini is still writing.

        True: the turn is dealt with (answered, or failed with an error the caller has been given).
        False: the streamed path broke before it had sent anything; answer the normal way instead."""
        from brain import TurnCancelled

        a = self.assistant
        shown = None if typed else heard
        reply = StreamingReply(
            a.tts.session(), _HttpTransport(self),
            lambda voice, text, chat_ms, tts_ms, actions: self._frame(shown, text, chat_ms, actions, voice, tts_ms,
                                                                      stt_ms),
            a.brain.actions, offset_ms=int((time.time() - self.t_start) * 1000),
            hold_s=config.STREAM_HOLD_MS / 1000.0, fillers=FILLERS if config.TOOL_FILLER else (),
            apology=ERROR_LINE)
        reply.start()
        try:
            try:
                result = a.think_stream(heard, self.music_state, reply)
            except TurnCancelled:
                print("[turn] the speaker hung up mid-reply")
                return True
            except TurnError as exc:
                reply.finish(error=exc)
                if not reply.decided:
                    raise                      # nothing was sent: the caller answers with the error
                reply.wait()
                print(f"[turn] failed after the reply had started: {exc}")
                return True
            except Exception as exc:
                reply.finish(error=exc)
                if reply.decided:
                    reply.wait()
                    print(f"[turn] the streamed reply broke after it had started: {type(exc).__name__}: {exc}")
                    return True
                a.stream_fails += 1
                off = "; streaming is now off until the server is restarted" if not a.stream_enabled() else ""
                print(f"[stream] broke before sending anything ({type(exc).__name__}: {exc}); "
                      f"answering the normal way{off}")
                return False
            reply.finish()
            reply.wait()
        finally:
            reply.close()

        if not reply.started and not reply.dead:       # nothing could be said, so the speaker gets the text
            raise TurnError(f"voice: {reply.voice_error or 'nothing was said'}", 502, reply=result["reply"],
                            action=result.get("actions") or {}, **({} if typed else {"heard": heard}))
        a.stream_fails = 0
        if reply.dead:
            print("[turn] the speaker hung up mid-reply")
        else:
            timing = result.get("timing") or {}
            _latency("typed" if typed else "spoken", heard, stt_ms, result, reply.tts_ms or 0,
                     reply.first_audio_ms or 0,
                     {"first_sentence_ms": timing.get("first_sentence_ms"), "ttft_ms": timing.get("ttft_ms"),
                      "hedged": timing.get("hedged"), "filler": reply.filler, "answer_ms": reply.answer_ms})
            print(f"[turn] {'typed' if typed else 'heard'} \"{heard}\" | "
                  f"{'' if typed else f'Whisper {stt_ms} ms | '}Gemini {result['ms']} ms {_tools(result)}| "
                  f"{reply.voice.name} {reply.tts_ms} ms (streamed)\nfriday> {result['reply']}{_action(result)}")
            _save_reply(bytes(reply.audio), reply.voice.fmt)
        a.after_turn(heard, result)
        return True

    def _chat(self, text: str) -> None:
        result = self.assistant.think(text, self.music_state)
        self._json({"reply": result["reply"], "ms": result["ms"], "tools": [t["tool"] for t in result["tools"]],
                    "action": result.get("actions") or {}})
        print(f"[turn] typed \"{text}\" -> {result['reply']}")
        self.assistant.after_turn(text, result)

    def _say(self, text: str) -> None:
        if self._wants_stream() and self._turn_streamed(text, 0, typed=True):
            return
        result = self.assistant.think(text, self.music_state)
        backend, first, rest, tts_ms = self._first_voice(result["reply"], action=result.get("actions") or {})
        header = self._header(None, result, backend, tts_ms, 0)
        audio = self._stream(header, first, rest, backend)
        _latency("typed", text, 0, result, tts_ms, self.first_audio_ms)
        print(f"[turn] typed \"{text}\" | Gemini {result['ms']} ms {_tools(result)}| {backend.name} {tts_ms} ms\n"
              f"friday> {result['reply']}{_action(result)}")
        _save_reply(audio, backend.fmt)
        self.assistant.after_turn(text, result)

    def _ask(self, body: bytes, debug: bool) -> None:
        a = self.assistant
        pcm = read_audio(body)
        _save("last_question.wav", pcm)
        heard, raw, stt_ms = a.hear(pcm)

        if heard and not debug and self._wants_stream() and self._turn_streamed(heard, stt_ms, typed=False):
            return

        result = {"reply": NOT_HEARD_REPLY, "ms": 0, "tools": [], "actions": {}}
        if heard:
            result = a.think(heard, self.music_state)

        if debug:
            self._json({"heard": heard, "raw_transcript": raw, "reply": result["reply"], "stt_ms": stt_ms,
                        "chat_ms": result["ms"], "tools": result["tools"],
                        "seconds": round(pcm.size / config.SAMPLE_RATE, 2)})
        else:
            backend, first, rest, tts_ms = self._first_voice(result["reply"], heard=heard,
                                                             action=result.get("actions") or {})
            header = self._header(heard, result, backend, tts_ms, stt_ms)
            audio = self._stream(header, first, rest, backend)
            _latency("spoken", heard, stt_ms, result, tts_ms, self.first_audio_ms)
            _save_reply(audio, backend.fmt)
            print(f"[turn] heard \"{heard or '(nothing clear)'}\" | Whisper {stt_ms} ms | Gemini {result['ms']} ms "
                  f"{_tools(result)}| {backend.name} {tts_ms} ms\nfriday> {result['reply']}{_action(result)}")
        a.after_turn(heard, result)


LATENCY_COLUMNS = ("time", "kind", "first_audio_ms", "stt_ms", "prompt_ms", "gemini_ms", "tool_ms", "tts_ms", "quick",
                   "stream", "first_sentence_ms", "ttft_ms", "hedged", "filler", "answer_ms", "said")


def _latency(kind: str, said: str, stt_ms: int, result: dict, tts_ms: int, first_audio_ms: int,
             stream: dict | None = None) -> None:
    """Where the time went, on the console and as one line in latency.log.
    `stream` is given for a reply that started while Gemini was still writing."""
    t = result.get("timing") or {}
    rounds = t.get("gemini_ms") or []
    gemini = "+".join(map(str, rounds)) or "0"
    s = stream or {}
    if stream is None:
        line = (f"[latency] {first_audio_ms} ms to first audio = hearing {stt_ms} + memory/prompt "
                f"{t.get('prompt_ms', 0)} + Gemini {gemini}"
                f"{' (quick reply)' if t.get('fast') else ''} + tools {t.get('tool_ms', 0)} + voice {tts_ms}")
    else:
        line = (f"[latency] {first_audio_ms} ms to first audio (streamed) = hearing {stt_ms} + Gemini "
                f"{s.get('first_sentence_ms')} to the first sentence (memory/prompt {t.get('prompt_ms', 0)}, "
                f"first word {s.get('ttft_ms')}{', retried' if s.get('hedged') else ''}) + voice {tts_ms}")
        if s.get("filler"):
            line += f"; said {s['filler']!r} while the tool ran"
        line += (f"; the answer itself from {s.get('answer_ms')} ms; "
                 f"Gemini {gemini} and tools {t.get('tool_ms', 0)} in all")
    print(line)
    if not config.LATENCY_LOG:
        return
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "first_audio_ms": first_audio_ms,
           "stt_ms": stt_ms, "prompt_ms": t.get("prompt_ms", 0), "gemini_ms": "+".join(map(str, rounds)),
           "tool_ms": t.get("tool_ms", 0), "tts_ms": tts_ms, "quick": int(bool(t.get("fast"))),
           "stream": int(stream is not None), "first_sentence_ms": s.get("first_sentence_ms"),
           "ttft_ms": s.get("ttft_ms"), "hedged": int(bool(s.get("hedged"))) if stream is not None else "",
           "filler": int(bool(s.get("filler"))) if stream is not None else "", "answer_ms": s.get("answer_ms"),
           "said": (said or "")[:80].replace("\t", " ")}
    try:
        path = HERE / "latency.log"
        header = "\t".join(LATENCY_COLUMNS)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if f.readline().rstrip("\n") != header:      # an older log with other columns: keep it aside
                    f.close()
                    path.replace(HERE / f"latency.{time.strftime('%Y%m%d-%H%M%S')}.log")
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(header + "\n")
            f.write("\t".join("" if row[c] is None else str(row[c]) for c in LATENCY_COLUMNS) + "\n")
    except OSError:
        pass


def _action(result: dict) -> str:
    act = result.get("actions")
    return f"\n[action] {json.dumps(act, ensure_ascii=False)}" if act else ""


def _bitrate_bytes() -> float:
    """Bytes per second of the converted songs (MUSIC_BITRATE like "64k")."""
    b = config.MUSIC_BITRATE.lower().rstrip("bps")
    try:
        return float(b[:-1]) * 1000 / 8 if b.endswith("k") else float(b) / 8
    except ValueError:
        return 8000.0


def _tools(result: dict) -> str:
    names = [t["tool"] for t in result.get("tools", [])]
    return f"(tools: {', '.join(names)}) " if names else ""


def _save_reply(audio: bytes, fmt: str) -> None:
    """last_reply.wav (local voice) or last_reply.mp3 (Edge voice)."""
    if not config.SAVE_AUDIO or not audio:
        return
    try:
        if fmt == "mp3":
            (HERE / "last_reply.mp3").write_bytes(audio)
        else:
            save_wav(HERE / "last_reply.wav", np.frombuffer(audio, dtype="<i2"), config.SAMPLE_RATE)
    except OSError:
        pass


def _save(name: str, pcm: np.ndarray) -> None:
    if config.SAVE_AUDIO and pcm.size:
        try:
            save_wav(HERE / name, pcm, config.SAMPLE_RATE)
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # The speaker hangs up whenever a song is skipped or paused: not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


# ====================================================================
#  Discovery: the S3 broadcasts "I2SONIC?", we answer "I2SONIC!<http port>"
# ====================================================================

def discovery_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", config.UDP_PORT))
    reply = f"I2SONIC!{config.HTTP_PORT}".encode()
    while True:
        try:
            data, addr = sock.recvfrom(64)
        except OSError:
            continue
        if data.startswith(b"I2SONIC?"):
            sock.sendto(reply, addr)
            print(f"[discovery] speaker at {addr[0]} found us")


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                # no packet is sent; just picks the LAN interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


# ====================================================================
#  Entry point
# ====================================================================

def run_test(a: Assistant, text: str | None, wav: str | None, music_state: str | None = None) -> None:
    if wav:
        pcm, rate = load_wav(wav)
        heard, raw, stt_ms = a.hear(resample(pcm, rate, config.SAMPLE_RATE))
        print(f"[stt] {stt_ms} ms: \"{raw}\"")
        text = heard or None
        if not text:
            print("[stt] nothing clear was heard")
            return
    result = a.think(text, music_state)
    print(f"[gemini] {result['ms']} ms {_tools(result)}\nfriday> {result['reply']}{_action(result)}")
    t0 = time.time()
    backend, first, rest, _ms = a.voice_start(result["reply"])
    audio = first + b"".join(rest)
    print(f"[tts] {int((time.time() - t0) * 1000)} ms with {backend.name}")
    name = "last_reply.mp3" if backend.fmt == "mp3" else "last_reply.wav"
    if backend.fmt == "mp3":
        (HERE / name).write_bytes(audio)
    else:
        save_wav(HERE / name, np.frombuffer(audio, dtype="<i2"), config.SAMPLE_RATE)
    print(f"[tts] wrote {HERE / name}; play it to hear Friday")
    a.after_turn(text, result)
    a.background.idle(60)


def main() -> None:
    ap = argparse.ArgumentParser(description="FRIDAY local server")
    ap.add_argument("--test", metavar="TEXT", help="ask one question and write last_reply.wav")
    ap.add_argument("--test-wav", metavar="FILE", help="run a recording through the whole pipeline")
    ap.add_argument("--music", nargs="?", const="state=stopped", metavar="STATE",
                    help='with --test: pretend the speaker can play music, e.g. "state=playing;track=12;pos=30;vol=70"')
    args = ap.parse_args()

    bad = config.problems()
    if bad:
        print("Can't start:\n  " + "\n  ".join(bad) + "\nCopy .env.example to .env and fill it in.")
        sys.exit(1)

    try:
        assistant = Assistant(with_stt=not args.test)
    except Exception as exc:
        print(f"Can't start: {exc}")
        sys.exit(1)

    if args.test or args.test_wav:
        try:
            run_test(assistant, args.test, args.test_wav, args.music)
        except TurnError as exc:
            print(f"[failed] {exc}")
        return

    Handler.assistant = assistant
    import gemini

    gemini.keepalive_start()
    import weather

    weather.start_prefetch(lambda: str(assistant.memory.profile().get("city") or ""))
    threading.Thread(target=discovery_loop, name="discovery", daemon=True).start()
    httpd = Server(("0.0.0.0", config.HTTP_PORT), Handler)
    print(f"\n[friday] listening on port {config.HTTP_PORT} (discovery on UDP {config.UDP_PORT})")
    for ip in lan_ips():
        print(f"         http://{ip}:{config.HTTP_PORT}/health")
    print("         Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[friday] stopping")


if __name__ == "__main__":
    main()
