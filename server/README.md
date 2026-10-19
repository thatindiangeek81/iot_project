# friday-local: Friday's brain on your laptop

> In the Friday repo this is the `server/` folder.

The laptop hears (faster-whisper), remembers (PostgreSQL), speaks (an Irish neural
voice online, or Piper / the Windows voice offline) and asks **Gemini** to think. It speaks the same protocol as the
Cloudflare Worker, so the S3 uses whichever one is available: the laptop when it's
on, the Worker when it's not.

```
S3 --(Wi-Fi, HTTP)--> friday-local --> faster-whisper   (hearing, on the laptop)
                                   --> PostgreSQL       (memory, on the laptop)
                                   --> Gemini API       (thinking, web search)
                                   --> Piper / Windows  (voice, on the laptop)
```

## What Friday can do now

| Say | What happens |
|---|---|
| "Remember that the spare key is in the kitchen drawer" | Saved to memory (`save_memory`) |
| "Where's the spare key?" | Answered from memory: related memories are put in her prompt automatically |
| "What did I tell you about my car?" | Searches memories and past conversations (`recall_memory`) |
| "Forget my favourite colour" | Removes that memory (`forget_memory`) |
| "My name is Deep" / "I live in Bengaluru" | Stored in your profile, which is in every prompt (`update_profile`) |
| "What's the gold price today?" / "Any news on..." | Live web search (`web_search`): Google and DuckDuckGo are asked at once, the first good answer wins |
| "How's the weather in Gandhinagar?" / "Will it rain tomorrow?" | Open-Meteo forecast (`get_weather`), free and usually under a second. No city = your profile's city |
| "Play Kesariya" / "Play some Arijit Singh" / "Play the Brahmastra songs" | Plays from your own songs (`play_music`), then keeps going with others |
| "Pause" / "Resume" / "Next song" / "Go back" / "Stop" / "Louder" / "Volume 40" | `music_control` |
| "Play Tum Hi Ho next" | Queues it after the current song |
| "I added some songs" | Indexes new files in the music folder (`check_new_music`) |
| Normal chat | She remembers the last 8 exchanges; after 10 minutes of silence a new conversation starts and the old one is summarised |

She also **learns by herself**: after each reply, a background job asks Gemini whether you
mentioned anything worth keeping ("I've started going to the gym at 6") and saves it.
That never slows a reply down. Set `AUTO_MEMORY=0` to keep only what you ask her to remember.

## 1. PostgreSQL (already running in Docker)

Your database runs in Docker from the repo's `db/` folder (`pgvector/pgvector:pg17`, user `friday`,
database `friday`, port 5432; the password is `FRIDAY_DB_PASSWORD` in `db\.env`). Make sure it's up before starting friday-local:

```
cd db
docker compose up -d
docker compose ps
```

`friday-pg` should say **running** (if not: `docker compose start`, or just open Docker Desktop).

You don't need to load any schema: friday-local creates its own tables (`profile`, `facts`,
`episodes`, `turns`, `items`, `music_tracks`, `settings`) the first time it starts.
**Don't also load the old Supabase schema into this database**: it uses the same table names
with different columns, and the two would clash.

## 2. Set up friday-local (once)

Create a virtual environment (or reuse one that already has faster-whisper):

```
cd server
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

In `.env`, fill in the first three lines:

| Setting | Value |
|---|---|
| `GEMINI_API_KEY` | Same key as your Worker |
| `DEVICE_TOKEN` | Copy from `DEVICE_TOKEN` in the firmware's `main\secrets.h` |
| `DATABASE_URL` | Replace `YOUR_DB_PASSWORD` with `FRIDAY_DB_PASSWORD` from `db\.env` |

**Allow it through Windows Firewall.** Open a terminal **as Administrator** and run:

```
netsh advfirewall firewall add rule name="FRIDAY http" dir=in action=allow protocol=TCP localport=8765 profile=private
netsh advfirewall firewall add rule name="FRIDAY discovery" dir=in action=allow protocol=UDP localport=5006 profile=private
```

These rules only apply on networks Windows treats as **Private**. A phone hotspot often
joins as Public: Settings → Network & internet → Wi-Fi → (your hotspot) → **Private network**.

## 3. Check it without the speaker

```
python server.py --test "Remember that my bike is red"
python server.py --test "What colour is my bike?"
```

The first prints `[tool] save_memory(...)`, the second answers "red". Both write
`last_reply.wav`; play it to hear the voice.

## 4. Run it

```
python server.py
```

You'll see `[memory] PostgreSQL ready`, `[tts] using ...`, `[stt] ready` and
`[friday] listening on port 8765`. Leave it running and power the S3: it prints
`[brain] found the laptop`. Every question then shows up here with timings:

```
[tool] save_memory({"fact": "The spare key is in the kitchen drawer."}) -> saved memory 3: ...
[turn] heard "Remember the spare key is in the kitchen drawer" | Whisper 640 ms | Gemini 1100 ms (tools: save_memory) | voice 280 ms
friday> Got it, boss. The spare key is in the kitchen drawer.
[memory] learned: saved memory 4: The user goes to the gym at 6 am.
```

To test from the laptop as if you were the S3: `python client.py --say "hello"` (or
`--ask question.wav`). It finds the server the same way the S3 does.

## Looking at and editing memory

Open **Adminer** at <http://localhost:8080> (System PostgreSQL, Server `db`, Username `friday`,
your password, Database `friday`). Click a table → **Select data** to browse, **edit** on a row to change
it, **New item** to add one. Changes apply on the next question, no restart.

| Table | What's in it |
|---|---|
| `profile` | One row; `content` is JSON, e.g. `{"name": "Deep", "city": "Bengaluru"}`. Always in her prompt |
| `facts` | Long-term memories. Add rows by hand too: just fill `content` and she'll index it. Set `is_active` false to hide one |
| `episodes` | One per conversation, with a summary |
| `turns` | Every question and answer, with which tools ran |
| `items` | Free-form storage (notes, lists...), for later features |
| `music_tracks` | Your music library: official titles, artists, albums and artwork from Apple, play counts. Fix a wrong match by editing `title` / `artist` |
| `settings` | `extra_instructions`: anything you want added to her personality, e.g. `"Call me Deep. Answer in Hinglish if I speak Hinglish."` |

## Settings worth knowing (`.env`)

| Setting | Does |
|---|---|
| `GEMINI_MODEL` | Default `gemini-3.5-flash-lite`, the same model as the Worker |
| `WEB_SEARCH` / `SEARCH_PROVIDER` | Live web search. `auto` (default) tries Google Search grounding first; on a key without billing Google refuses, and Friday uses **DuckDuckGo** instead (free, no key) for the next hour. `duckduckgo` skips Google entirely. `SEARCH_REGION=in-en` gives Indian results |
| `AUTO_MEMORY` | `1` = learn facts by herself after each reply (one small extra Gemini call, in the background) |
| `HISTORY_TURNS` / `SESSION_GAP_MIN` | How much recent conversation she keeps word for word, and how long a silence starts a new one |
| `WHISPER_MODEL` | `base` is fast; `small` hears a noisy mic better but is ~3x slower on CPU |
| `WHISPER_DEVICE` | `cpu` works everywhere. `cuda` + `WHISPER_COMPUTE=int8_float16` uses your Quadro (needs CUDA 12 cuBLAS and cuDNN 9 installed); if CUDA fails it falls back to the CPU by itself |
| `TTS_ENGINE` / `EDGE_VOICE` | `edge` (default) speaks with Microsoft's Irish female neural voice `en-IE-EmilyNeural` over the internet, through the free Edge read-aloud service (unofficial: Microsoft could change or block it). If it fails, the reply is spoken by the local voice instead. `EDGE_RATE=+10%` speeds her up. `local` = offline voices only |
| `SAPI_VOICE` | Windows voice for the local backup, default `Zira` (female). `python tts.py` lists what you have |
| `PIPER_EXE` / `PIPER_MODEL` | Set both for a much nicer voice than the Windows one. Get a Piper Windows build and an `.onnx` voice such as `en_US-amy-medium` (plus its `.json`) |

## Music

Friday plays songs from your own folder (`MUSIC_DIR`, now `D:\songs`). The laptop keeps the
library and the queue; the speaker streams one song at a time and mixes it under her voice
(quieter while you talk to her, back up after).

**Once:** install ffmpeg (it converts songs for the speaker and makes M4A files playable),
then index the folder:

```
winget install Gyan.FFmpeg
python music.py scan
```

Open a **new** terminal after installing ffmpeg so it's on the PATH. The scan looks each new
song up on Apple's iTunes Search API (free, ~20 lookups a minute), then converts every song
once into `music_cache\` (16 kHz mono, the speaker's format; about 2 MB per song). Later
scans only touch new or changed files, and a renamed file keeps its metadata.

**New songs:** drop them in the folder. Friday checks every `MUSIC_SCAN_MIN` minutes (30 by
default), or say "I added some songs". `python music.py list --review` shows matches worth a look.

| Setting | Does |
|---|---|
| `MUSIC_DIR` | The folder with your songs |
| `MUSIC_SCAN_MIN` | How often to look for new songs (0 = only when you ask) |
| `MUSIC_AUTOPLAY` | `1` = after what you asked for, keep playing other songs (artist first, then a shuffle) |
| `MUSIC_LOUDNORM` | `1` = even out loud and quiet songs |
| `FFMPEG` | Full path to `ffmpeg.exe` if it isn't on the PATH |

Check it without the speaker: `python server.py --test "play kesariya" --music` prints the
`[action]` the speaker would get, and `python client.py --song 12` downloads song 12 exactly as
the speaker receives it (`song.mp3`).

Music needs the laptop: when the S3 falls back to the Cloudflare Worker, Friday says she
can't play music right now.

## Speed

Every question prints where its time went, and the same numbers go into `latency.log`
(tab-separated, opens in Excel):

```
[latency] 2140 ms to first audio = hearing 610 + memory/prompt 12 + Gemini 980 + tools 0 + voice 520
```

That's measured on the laptop, from the moment the recording arrives until the first audio leaves.
The S3's `[time]` line adds the upload and the speaker's buffer on top.

| Setting | Effect |
|---|---|
| `GEMINI_REASONING=low` | Gemini thinks briefly before answering (Gemini 3 can't skip thinking entirely). `medium` or `high` for harder questions, at a cost in speed. If the model refuses the setting, Friday notices and uses Google's default |
| `MEMORY_ALL_FACTS_MAX=40` | While she knows fewer than 40 things about you, they all go in her prompt and no memory search runs (that search cost a trip to Google on every question) |
| `TTS_CACHE_MAX=500` | Short lines she says often ("Paused.", "Volume 50 percent.") are kept as audio in `voice_cache\`, so they play at once instead of waiting 1–3 s for Microsoft. The common ones are made in the background when the server starts |
| `FAST_MUSIC_REPLIES=1` | "Pause", "next song", "play X" are answered by the laptop ("Here's Kesariya.") instead of a second Gemini call |
| `WHISPER_DEVICE=cuda` + `WHISPER_COMPUTE=int8_float16` | Hearing on the Quadro GPU instead of the CPU, usually 3–4x faster. Needs the CUDA 12 cuBLAS and cuDNN 9 libraries; if they're missing, the server says so and stays on the CPU |

The database isn't a bottleneck: all of its queries together take about 10–20 ms per question,
and memories and the music library are kept in RAM.

## Privacy

Recordings, transcripts and memories stay on this laptop. What leaves it: the text of
each question (plus the profile and relevant memories) goes to Gemini, and each memory is
sent once to Gemini's embedding API to index it. Passwords, PINs and card numbers are refused.

## If something fails

| Problem | Fix |
|---|---|
| `can't connect to PostgreSQL ... password authentication failed` | Wrong password in `DATABASE_URL`: use the one Adminer accepts |
| `can't connect to PostgreSQL ... Connection refused` | Docker Desktop isn't running, or the container is stopped: `cd db` then `docker compose start` |
| `GEMINI_API_KEY is not set` | `.env` missing or in the wrong folder (it must sit next to `server.py`) |
| `gemini 400/404 ... model` | Model name wrong for your key; try `GEMINI_MODEL=gemini-3.5-flash` |
| `[search] Google grounding refused ...` | Normal on a free key: searches go to DuckDuckGo. Set `SEARCH_PROVIDER=duckduckgo` to skip the Google attempt |
| `[tool] web_search(...) -> {"error": ...duckduckgo...}` | `pip install ddgs`, or DuckDuckGo is rate-limiting you: wait a minute |
| `time zone Asia/Kolkata not found` | `pip install tzdata` |
| The S3 never finds the laptop | Firewall rules above, network set to Private, same Wi-Fi. Check with `python client.py --find` from another PC |
| First answer slow | Whisper loads on start; the first Gemini call also opens its connection |
| `[music] 0 songs in the library` | Run `python music.py scan` |
| `ffmpeg not found` at start-up | `winget install Gyan.FFmpeg`, open a new terminal; or set `FFMPEG=C:\path\to\ffmpeg.exe` in `.env`. Without it MP3s still play, M4As don't |
| She plays the wrong song | Check the match with `python music.py list`; fix `title` / `artist` in Adminer |
