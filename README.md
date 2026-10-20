# Friday

A DIY smart speaker with an Iron Man–style assistant. Hold the BOOT button on the ESP32-S3,
ask something, and Friday answers out loud. Your laptop does the hearing (faster-whisper), the
thinking (Gemini), the remembering (PostgreSQL) and the voice (an Irish neural voice); the S3 just
listens and plays. If the laptop is off, a Cloudflare Worker answers instead (no memory or music).

```
ESP32-S3 ── Wi-Fi ──▶ server/   laptop: Whisper · Gemini · memory · voice · music
    │
    └── laptop off ──▶ worker/   Cloudflare Worker: Whisper · Gemini · voice
```

| Folder | What it is |
|---|---|
| [`firmware/`](firmware) | ESP32-S3 firmware (ESP-IDF 5.5): mic, speaker mixer, Wi-Fi, laptop discovery, music, offline SAM voice |
| [`server/`](server) | The laptop brain in Python: hearing, Gemini + tools, memory, voice, music library |
| [`worker/`](worker) | Cloudflare Worker used as the backup brain |
| [`db/`](db) | PostgreSQL + Adminer in Docker: Friday's memory |
| [`cortex-web/`](cortex-web) | Cortex upload side, hosted on Vercel: upload page + ingest (parse, chunk, embed into Supabase pgvector). See [`docs/cortex.md`](docs/cortex.md) |
| [`Cortex/`](Cortex) | The original Cortex RAG project; its parser was reused in `cortex-web/`. Friday no longer calls it |
| [`docs/`](docs) | Architecture overview and [why replies take the time they do](docs/latency.md) |

## Getting started

1. **Database**: `cd db`, copy `.env.example` to `.env`, set a password, `docker compose up -d`.
2. **Server**: follow [`server/README.md`](server/README.md): venv, `pip install -r requirements.txt`,
   fill in `server/.env`, `python server.py`.
3. **Firmware**: copy `firmware/main/secrets.example.h` to `secrets.h`, fill it in, then build and flash
   as in [`firmware/README.md`](firmware/README.md).
4. **Worker** (optional backup brain): see [`worker/README.md`](worker/README.md).

## Secrets

Nothing private is committed. `server/.env`, `db/.env`, `worker/.dev.vars` and `firmware/main/secrets.h`
are git-ignored, and each has an `.example` template beside it. `DEVICE_TOKEN` must be the same in the
firmware, the server and the Worker.
