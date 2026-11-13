# Changelog

## 0.3.0

### Local-first

- **jina-embeddings-v4 runs locally on Apple Silicon** through Jina's official 8-bit MLX port, pinned to a revision, with no PyTorch. It matches the hosted API at cosine 0.999 for text and 0.988–0.995 for frames, so both backends share the model name and indexes are interchangeable. A full index of a 6½-minute video takes 27.5 s with no API keys, against 8–12 minutes on the rate-limited API.
- `DVQA_EMBED_BACKEND=auto` is the new default: MLX when available, otherwise the Jina API.
- New `fetch-models` command; `doctor` and `video_status` report where embeddings run and whether the weights are cached.
- The published MLX port caches results by Python object id, frees its vision weights after the first image, and mishandles padding in its compiled text path. The server bypasses all three, and Apple-only tests pin each failure.

### Verification

- `video_ask(verify=true)` uses Jev 1.14 (TypeSafe) to judge whether each moment's transcript answers the question and whether each frame shows it, then reranks by support. The result adds `evidence.supported`, and a frames-only rescue runs when nothing is supported. Each moment reports `transcript_answers`, `support` and each frame's `shows`.
- Off by default (it sends excerpts and frames to TypeSafe). Enable it per call, or with `DVQA_VERIFY=jev`. New error codes: `verify_unavailable` and `verify_failed`.


## 0.2.0

### Interface

- Every tool now declares a typed result, published as `outputSchema`, validated before sending and returned as `structuredContent`. Tools that return images (`video_ask`, `video_frames`, `video_chapters`) send the same structure followed by caption and image pairs.
- All 30 error codes are registered with a category, a hint and a retryable flag. Error text has a fixed format: `Error executing tool <tool>: <code>: <message> — <hint>`.
- `docs/tool-contract.md` is generated from the server (`distributed-video-qa-mcp contract`), and a test fails when it is stale.
- Every argument and result field carries a description; a test enforces it.

### Added

- `video_ingest_batch`: many sources in one call; playlist and channel URLs expand to their first N videos; bad sources are reported in `skipped` without stopping the rest.
- `video_find_phrase`: every occurrence of exact words, in time order.
- `video_ask` now reports `evidence` (best transcript and frame similarity, keyword match count), so a weak match can be told from a strong one.
- MCP resources: `video://library`, `video://{id}/transcript`, `video://{id}/subtitles.srt`, `video://{id}/info`, `video://{id}/frame/{seconds}`.
- MCP prompts: `video_brief`, `video_fact_check`, `video_study_notes`, `video_compare`, `video_find_moment`.
- `video_status` reports `ready` and a `problems` list, and `doctor` exits non-zero when not ready.
- Operations guide: deployment, state layout, backups, capacity and cost, troubleshooting.

### Changed

- Keyword search ignores stopwords, so `keyword_matches` reflects content words only.
- The server is split into `app` (shared state), `server` (tools), `resources` and `prompts`.

## 0.1.0

- First release: link-to-file download with audio merge (H.264/AAC preferred), Whisper transcription (MLX, faster-whisper or a chunked parallel API), parallel frame sampling with scene cuts and de-duplication, `jina-embeddings-v4` for transcript and frames, hybrid retrieval with weighted reciprocal-rank fusion, resumable jobs, SSRF and path guards, and bearer-token HTTP.
