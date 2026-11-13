[![Distributed Systems — Any video. Every frame. Answered.](docs/assets/hero.png)](https://distributed.systems)

# Distributed Video QA MCP

**Give your MCP client a video link. It downloads it with sound, transcribes it, indexes every scene, and answers questions with timestamps and the actual frames.**

[![Verify](https://github.com/arthurcolle/distributed-video-qa-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurcolle/distributed-video-qa-mcp/actions/workflows/ci.yml)
**v0.1.0 · 17 tools · Python 3.11+ · MIT**

Try asking your MCP client:

> Index https://www.youtube.com/watch?v=… and tell me every claim the speaker makes about pricing, with timestamps.

> Find the moment where the architecture diagram is on screen and read me the labels.

> Download this link at 1080p with audio into ~/Movies.

![Fetch, listen, see, ask: the pipeline from a link to answers with timestamps and frames.](docs/assets/workflow.svg)

## What it does

- **Link to file, with sound.** Above 360p most sites serve video and audio as separate streams. `video_download` picks the quality you ask for, fetches both in parallel fragments and muxes them into one MP4, MKV or WebM. `video_probe` shows the quality menu first, flagging which qualities have no sound of their own. Audio-only export is also supported (m4a, mp3, opus, flac, wav). Downloads go through [yt-dlp](https://github.com/yt-dlp/yt-dlp), so any site it supports works.
- **Transcript.** Whisper large-v3-turbo runs on-device through MLX on Apple Silicon or faster-whisper elsewhere. Without a local model it uses any OpenAI-compatible endpoint (Groq, OpenAI). API transcription cuts the audio on silences and sends the chunks concurrently.
- **Frames.** Frames are sampled every few seconds plus at every scene cut. The timeline is decoded in parallel FFmpeg segments. Near-duplicate frames are removed with a perceptual hash and a colour check, so a talking head doesn't use up the frame budget.
- **One embedding space for words and pictures.** Transcript passages and frames are both embedded with [`jina-embeddings-v4`](https://jina.ai/embeddings/). A text question can then find a frame directly ("the slide with the pricing table") without a captioning step.
- **Hybrid retrieval.** Three channels run together: semantic search over the transcript, BM25 keyword search (SQLite FTS5), and text-to-image search over frames. They are fused by weighted reciprocal rank so that speech and picture count equally. Every hit says which channel matched it.
- **Evidence, not a second model's opinion.** `video_ask` returns the relevant moments as timestamped transcript plus the frames as images. Your client's model reads the screen itself and answers, and nothing else is paid to summarise along the way.

## Install

Needs FFmpeg on `PATH` and a [Jina API key](https://jina.ai/api-dashboard/) (`JINA_API_KEY`).

```sh
git clone https://github.com/arthurcolle/distributed-video-qa-mcp.git
cd distributed-video-qa-mcp
uv venv && uv pip install -e ".[mlx]"            # Apple Silicon: on-device Whisper
# or: uv pip install -e ".[faster-whisper]"      # Linux / CUDA / CPU
# or: uv pip install -e .  and set GROQ_API_KEY  # transcription over an API
.venv/bin/distributed-video-qa-mcp doctor
```

`doctor` prints what it found and exits non-zero if something required is missing.

### Connect

Claude Code:

```sh
claude mcp add video-qa -e JINA_API_KEY=… -- /absolute/path/distributed-video-qa-mcp/.venv/bin/distributed-video-qa-mcp
```

Any MCP host (stdio):

```json
{
  "mcpServers": {
    "video-qa": {
      "command": "/absolute/path/distributed-video-qa-mcp/.venv/bin/distributed-video-qa-mcp",
      "env": { "JINA_API_KEY": "…" }
    }
  }
}
```

Shared server over streamable HTTP:

```sh
DVQA_HTTP_TOKEN=$(openssl rand -hex 32) distributed-video-qa-mcp serve --http --host 0.0.0.0 --port 4005 \
  --allowed-host video-qa.internal:4005
```

Clients send `Authorization: Bearer <token>` to `/mcp`. `/health` is open. Binding to a non-loopback address without a token is refused.

## Tools

| Tool | Purpose |
| --- | --- |
| `video_probe` | Title, length, and the quality menu for a link, with sizes and whether each needs an audio merge |
| `video_download` | Save a link as video with audio merged, video only, or audio only; returns a job |
| `video_ingest` | Download (or read a local file), transcribe, sample frames, embed; returns a job |
| `video_job` / `video_jobs` / `video_cancel` | Wait on, list, and stop jobs, with progress notifications |
| `video_ask` | Moments relevant to a question: transcript excerpts with timestamps plus frames as images |
| `video_search` | Ranked hits over what was said and what was shown, with time and video filters |
| `video_search_by_image` | Frames that look like a picture you supply, and passages about it |
| `video_transcript` | The transcript or a range of it as timestamped text, plain, SRT, WebVTT or JSON |
| `video_frames` | Frames at exact times or across a range, decoded fresh at full resolution for reading text |
| `video_chapters` | Uploader chapters, or topic boundaries found from the embeddings, with keywords and a frame each |
| `video_clip` | Export a time range to MP4, stream-copied or frame-accurate |
| `video_list` / `video_info` / `video_delete` | Manage the index |
| `video_status` | Backends, versions, limits and index size |

Long work runs as background jobs. Tools accept `wait_seconds` (up to 300), so short videos usually finish within one call. Re-ingesting a video reuses every stage that already completed with the same settings, and a failed or interrupted job resumes where it stopped.

The same pipeline is available from the terminal:

```sh
distributed-video-qa-mcp ingest https://… --quality 720p
distributed-video-qa-mcp search "where does he show the order book"
```

## Configuration

All settings are environment variables. The defaults suit a laptop.

| Variable | Default | Meaning |
| --- | --- | --- |
| `JINA_API_KEY` / `DVQA_EMBED_API_KEY` | — | Embedding key |
| `DVQA_EMBED_MODEL`, `DVQA_EMBED_DIM` | `jina-embeddings-v4`, `1024` | Model and Matryoshka dimension |
| `DVQA_EMBED_TPM` | `90000` | Token-per-minute budget; set it to your Jina tier |
| `DVQA_EMBED_IMAGE_PX` | `448` | Long side of frames sent for embedding (≈1,440 tokens each; 640px ≈ 2,990) |
| `DVQA_TRANSCRIBE_BACKEND` | `auto` | `mlx`, `faster-whisper`, `api`, or `none` for frames only |
| `DVQA_WHISPER_MODEL` | per backend | e.g. `mlx-community/whisper-large-v3-turbo` |
| `DVQA_TRANSCRIBE_BASE_URL`, `DVQA_TRANSCRIBE_API_KEY` | Groq or OpenAI if their key is set | Any OpenAI-compatible transcription endpoint |
| `DVQA_FRAME_INTERVAL` | `5` | Seconds between sampled frames (scene cuts are added) |
| `DVQA_MAX_FRAMES` | `600` | Frames kept per video after de-duplication |
| `DVQA_CPU_WORKERS` | half the cores, 2–8 | Parallel FFmpeg decoders and image workers |
| `DVQA_MAX_CONCURRENT_JOBS` | `2` | Jobs running at once; more are queued |
| `DVQA_INGEST_HEIGHT` | `720` | Download height for indexing |
| `DVQA_MAX_DURATION`, `DVQA_MAX_FILESIZE` | 6 h, 8 GiB | Refuse anything larger |
| `DVQA_STATE_DIR` | `~/.local/state/distributed-video-qa-mcp` | Index, media and downloads |
| `DVQA_ALLOWED_ROOTS` | home and tmp (stdio); none (HTTP) | Where local files may be read and written |
| `DVQA_COOKIES_FILE` | — | Netscape cookie file for sites that require sign-in |
| `DVQA_ALLOW_PRIVATE_HOSTS` | off | Permit URLs that resolve to private or loopback addresses |
| `DVQA_HTTP_TOKEN` | — | Bearer token for HTTP mode |

## How it works

```
source ─ probe ─┬─ audio ─ Whisper (local, or chunked on silence and sent concurrently) ─ passages ─ text vectors ─┐
                │                                                                                               ├─ SQLite index
                └─ frames (parallel FFmpeg segments, interval + scene cut) ─ dedupe ─ image vectors ───────────────┘
```

Both branches run at the same time. The index is a single SQLite file in WAL mode. It holds transcript segments, about 30-second overlapping passages, frames, an FTS5 table, and float32 vectors. Search loads each vector set into one NumPy matrix and scores it exactly with a single matrix product, which stays well under a second into the hundreds of thousands of vectors on one machine.

The Jina client keeps a sliding one-minute token budget. It learns the real per-image cost from usage reports, so a large ingest runs at your rate limit instead of failing on it.

## Safety

- URLs must be plain http(s) to public addresses. Loopback, private, link-local and cloud-metadata addresses are refused unless you opt in, as are embedded credentials and non-HTTP schemes. Name resolution happens before the download starts. Against a hostile DNS server, run with egress controls.
- Local paths are resolved and must sit under `DVQA_ALLOWED_ROOTS`. In HTTP mode nothing outside the state directory is reachable unless you configure it.
- `video_delete` removes only index rows and the server's own copies. A file you ingested by path is never deleted.
- Logs go to stderr; stdout carries the MCP protocol.

Downloading is subject to each site's terms and to copyright law. Use this on media you have the right to copy.

## Development

```sh
uv pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pytest
```

The tests build a synthetic video with FFmpeg and use a deterministic offline embedder and a scripted transcriber, so no network access or keys are needed. They drive the whole pipeline through a real MCP client.

## License

MIT. The Distributed Systems name and wordmark are not covered; see [LICENSE](LICENSE).
