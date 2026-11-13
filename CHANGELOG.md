# Changelog

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
