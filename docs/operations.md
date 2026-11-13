# Operations

How to run, size, tune, back up and troubleshoot the server. For the tool interface see [tool-contract.md](tool-contract.md).

## Deployment modes

| Mode | Use when | Start |
| --- | --- | --- |
| **stdio** (default) | One user, the MCP host launches the server | `distributed-video-qa-mcp` |
| **Streamable HTTP** | Several clients share one index, or the GPU box is elsewhere | `distributed-video-qa-mcp serve --http --host 0.0.0.0 --port 4005` |

In HTTP mode:

- Set `DVQA_HTTP_TOKEN` (32+ random bytes). Clients send `Authorization: Bearer <token>`. Binding a non-loopback address without one is refused.
- Pass `--allowed-host <host:port>` for each public name, to enable DNS-rebinding protection.
- Terminate TLS in front (Caddy, nginx, a cloud load balancer). `/health` is unauthenticated and returns version and index counts for probes.
- `DVQA_ALLOWED_ROOTS` defaults to nothing, so clients can only ingest URLs and read the server's own media. Add a shared media folder explicitly if you want path ingest.

A minimal systemd unit:

```ini
[Unit]
Description=Distributed Video QA MCP
After=network-online.target

[Service]
User=videoqa
Environment=JINA_API_KEY=…
Environment=DVQA_HTTP_TOKEN=…
Environment=DVQA_STATE_DIR=/var/lib/videoqa
Environment=DVQA_MAX_CONCURRENT_JOBS=4
ExecStart=/opt/videoqa/.venv/bin/distributed-video-qa-mcp serve --http --host 127.0.0.1 --port 4005
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Keep secrets in an `EnvironmentFile` with mode 0600 rather than in the unit itself.

## State

Everything lives under `DVQA_STATE_DIR` (default `~/.local/state/distributed-video-qa-mcp`, created 0700):

```
index.sqlite3            videos, jobs, segments, passages, frames, FTS5 table, float32 vectors (WAL mode)
media/<video_id>/source/ downloaded media for URL ingests
media/<video_id>/frames/ kept frames, named <index>_<milliseconds>.jpg
downloads/               video_download output and clips/ for video_clip
```

Local files you ingest by path are read in place, never copied or deleted.

**Backup.** The index is a single SQLite file in WAL mode. Copy it consistently while the server runs with `sqlite3 index.sqlite3 ".backup /backups/index.sqlite3"`, then copy `media/`. A restored index whose media is missing still answers transcript questions; frame tools will report `source_missing` or a missing image for those videos.

**Upgrades.** Every stage stores a signature of the settings that produced it. After an upgrade or a settings change, re-ingest a video to recompute only the stages whose inputs changed. Changing the embedding model or dimension re-embeds both branches. Changing frame settings or `DVQA_EMBED_IMAGE_PX` redoes frames only.

## Local models (Apple Silicon)

- `distributed-video-qa-mcp fetch-models` downloads `jinaai/jina-embeddings-v4-mlx-8bit` (3.9 GB) and the base model's tokenizer and preprocessing files, each pinned to a commit. They live in the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`); delete that folder to reclaim the space.
- The embedding model loads on first use (about 1 s once cached) and stays resident, using about 5 GB of unified memory. MLX Whisper keeps its own model resident as well. On a 16 GB Mac, keep `DVQA_MAX_CONCURRENT_JOBS=1`.
- All MLX work runs on one thread per model. Concurrent jobs queue for it rather than contending for the GPU.
- `doctor` reports whether the weights are cached. Without them, the first ingest downloads them before embedding starts.

## Verification (Jev)

- Install the `jev` extra and set `TYPESAFE_API_KEY`. `video_status.verification` shows whether it is available and whether it runs by default (`DVQA_VERIFY=jev`).
- Cost per verified `video_ask`: one judgement per candidate moment (max_moments + 2) plus one per attached frame, and up to six more in a frames-only rescue. That is typically 10–20 calls, ~1,100 input tokens each, and 0.7–0.9 s, since calls run eight at a time.
- If TypeSafe fails mid-call, the answer still returns unverified, with `evidence.verify_error` set. If verification isn't configured at all, `verify=true` fails with `verify_unavailable`.
- Data sent: the question, the transcript lines of each candidate moment, and the candidate frames (JPEG, ≤768 px). Nothing is sent unless verification is requested.

## Capacity and cost

Measured on an Apple M4 Max with the defaults:

| Resource | Per hour of video | Notes |
| --- | --- | --- |
| Full local ingest (MLX Whisper + MLX embeddings) | ~4–5 min | 27.5 s for a 6:36 video on an M4 Max |
| Transcription (MLX large-v3-turbo) | ~80 s | 47× realtime here; other backends depend on hardware or the provider |
| Frame sampling | ~15 s | Scales with `DVQA_CPU_WORKERS` |
| Frames kept | ~700 at a 5 s interval, before de-duplication | Capped by `DVQA_MAX_FRAMES` (default 600 per video) |
| Embedding, local MLX | ~100 s for 600 frames (~160 ms each) | No cost; bound by the GPU |
| Embedding, Jina API | ~0.9 M tokens for 600 frames at 448 px, plus ~25 k for the transcript; ~10 min at 90 k tokens/min | Images dominate; cost scales with pixel area |
| Disk | ~60 MB of frames plus the source (720p ≈ 0.5–1 GB/h) | Delete with `video_delete` |
| Index RAM | 4 KB per vector at 1,024 dimensions | 300,000 vectors ≈ 1.2 GB, 35 ms per query |

Levers, in order of effect:

1. **Backend.** On Apple Silicon the local MLX embedder removes the token budget entirely. With the Jina API, set `DVQA_EMBED_TPM` to your tier's tokens-per-minute limit; it is then the main bound on ingest throughput.
2. **`DVQA_EMBED_IMAGE_PX`.** 448 px costs ~1,440 tokens per frame and 640 px ~2,990. Frames returned to the model are always full resolution, so this only affects retrieval.
3. **`DVQA_FRAME_INTERVAL` and `DVQA_MAX_FRAMES`.** For lectures and screencasts, a longer interval loses little because scene cuts are always kept. For fast-cut footage, keep it short.
4. **`DVQA_EMBED_DIM`.** 512 halves index RAM and query time with a small recall cost. Changing it re-embeds on the next ingest.
5. **`DVQA_MAX_CONCURRENT_JOBS`.** Raise it when transcription goes over an API. Local Whisper runs one inference at a time regardless.

## Observability

- `video_status` (or `distributed-video-qa-mcp doctor`) reports readiness, the active backends, the JavaScript runtimes, the limits and the index size. `ready: false` comes with reasons.
- `video_jobs(active_only=true)` lists running work with its stage and progress. `video_job(wait_seconds=…)` streams MCP progress notifications.
- Logs go to stderr (stdout carries the protocol in stdio mode). Use `--log-level INFO` or `DEBUG` to see yt-dlp and provider retries.
- `/health` in HTTP mode returns `{"ok": true, "version": …, "videos": …, "vectors": …}`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| YouTube offers only 144p–360p | No JavaScript runtime for the signature challenge | `video_status.js_runtimes` must not be empty. The default install includes Deno; otherwise install deno, node or bun |
| `probe_failed: Sign in to confirm you're not a bot` | Datacenter IP or age gate | Export cookies (Netscape format) and set `DVQA_COOKIES_FILE`, or ingest from a residential network |
| `embed_failed … 429` | Jina rate limit | Lower `DVQA_EMBED_TPM` below your tier; re-run the ingest, since finished stages are kept |
| Ingest slow at "embedding frames" | Token budget, not CPU | Raise your Jina tier and `DVQA_EMBED_TPM`, or lower `DVQA_EMBED_IMAGE_PX` / raise `DVQA_FRAME_INTERVAL` |
| `no_transcriber` | No local Whisper and no API key | Install the `mlx` or `faster-whisper` extra, set `GROQ_API_KEY`, or ingest with `transcribe=false` |
| `private_host` on an internal media server | SSRF guard | Set `DVQA_ALLOW_PRIVATE_HOSTS=1` on trusted deployments only |
| `path_not_allowed` | Path outside allowed roots | Add the folder to `DVQA_ALLOWED_ROOTS` (`:`-separated) |
| Job shows `interrupted` | The server restarted mid-job | Run the same ingest again; completed stages are reused |
| `verify_unavailable` | Jev not configured | `uv pip install -e ".[jev]"` and set `TYPESAFE_API_KEY` |
| First query after start takes ~2 s | The MLX embedding model is loading | Expected once per process |
| `jina-embeddings-v4 MLX weights are not available` | Weights not downloaded and offline | Run `distributed-video-qa-mcp fetch-models` while online |
| `video_ask` returns weak evidence | The video does not cover it, or the wording differs | Check `evidence.keyword_matches` and the similarities; try `video_search` with other words or `video_find_phrase` |
