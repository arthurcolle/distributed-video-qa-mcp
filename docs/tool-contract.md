# Tool contract

_Generated from the server definition (v0.3.0) by `distributed-video-qa-mcp contract`. Do not edit by hand; `tests/test_contract_doc.py` fails when this file is stale._

## Conventions

- **Typed results.** Every tool publishes an `outputSchema` and returns matching `structuredContent`, also serialized as the first text block. Results are validated against the schema before they are sent.
- **Images.** `video_ask`, `video_frames` and `video_chapters` follow the JSON with caption + JPEG pairs, in the order the structured result lists them.
- **Jobs.** Ingest and download return a `Job` at once. `wait_seconds` (≤ 300) blocks until it finishes or the time is up, streaming MCP progress notifications meanwhile; then poll `video_job`.
- **Errors.** A failed call returns `isError: true` with the text `Error executing tool <tool>: <code>: <message> — <hint>`, plus ` Retryable.` when retrying can succeed. Codes are listed [below](#error-codes).
- **Times** are seconds as numbers (`start`, `t`) with an `at`/`span` rendering as `m:ss` or `h:mm:ss`.

## Tools

| Group | Tool | Purpose | Hints |
| --- | --- | --- | --- |
| Acquire | [`video_probe`](#video_probe) | List download options for a URL | read-only · network |
| Acquire | [`video_download`](#video_download) | Download a video or its audio | writes · network |
| Index | [`video_ingest`](#video_ingest) | Index a video for questions | writes · network |
| Index | [`video_ingest_batch`](#video_ingest_batch) | Index several videos or a playlist | writes · network |
| Jobs | [`video_job`](#video_job) | Check a job | read-only · local |
| Jobs | [`video_jobs`](#video_jobs) | List jobs | read-only · local |
| Jobs | [`video_cancel`](#video_cancel) | Cancel a job | writes · local |
| Retrieve | [`video_ask`](#video_ask) | Gather evidence for a question | read-only · local |
| Retrieve | [`video_search`](#video_search) | Search inside videos | read-only · local |
| Retrieve | [`video_search_by_image`](#video_search_by_image) | Find frames that look like an image | read-only · local |
| Retrieve | [`video_find_phrase`](#video_find_phrase) | Find every time a phrase is said | read-only · local |
| Inspect | [`video_transcript`](#video_transcript) | Read the transcript | read-only · local |
| Inspect | [`video_frames`](#video_frames) | Look at frames | read-only · local |
| Inspect | [`video_chapters`](#video_chapters) | Outline a video | read-only · local |
| Inspect | [`video_clip`](#video_clip) | Export a clip | writes · local |
| Library | [`video_list`](#video_list) | List indexed videos | read-only · local |
| Library | [`video_info`](#video_info) | Describe one video | read-only · local |
| Library | [`video_delete`](#video_delete) | Remove a video from the index | writes · destructive · idempotent · local |
| Library | [`video_status`](#video_status) | Server status | read-only · local |

## Acquire

Turn a link into a file.

### `video_probe`

**List download options for a URL** · read-only · network · returns **`ProbeResult`**

Show what a link offers before downloading: title, length, and each quality with its size and whether it has sound on its own or needs the audio track merged in (most qualities above 360p do).

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `url` | string | **required** |  | Public http(s) link to a single video. |
| `all_formats` | bool | `False` |  | Also return every raw stream. |

Returns `ProbeResult`:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string \| null | The site's id for the video. |
| `extractor` | string \| null | yt-dlp extractor that handled the URL, such as 'Youtube'. |
| `title` | string \| null | Video title. |
| `uploader` | string \| null | Channel or uploader name. |
| `upload_date` | string \| null | YYYYMMDD. |
| `duration` | number \| null | Length in seconds. |
| `webpage_url` | string \| null | Canonical page URL. |
| `thumbnail` | string \| null | Thumbnail image URL. |
| `is_live` | bool | Live streams cannot be ingested. |
| `qualities` | Quality[] | Best first, one entry per available height. |
| `audio_tracks` | StreamFormat[] | Best audio streams first. |
| `formats` | StreamFormat[] \| null | Every raw stream, when all_formats is set. |

### `video_download`

**Download a video or its audio** · writes · network · returns **`Job`**

Save a link to a file. Fetches video and audio streams in parallel fragments and muxes them into one playable file, preferring H.264/AAC for MP4 so it opens everywhere. Returns a Job whose result is a DownloadReport.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `url` | string | **required** |  | Public http(s) link to a single video. |
| `quality` | string | `'best'` |  | 'best' or a maximum height such as '1080p' or '720p'. |
| `mode` | `video+audio` \| `video_only` \| `audio_only` | `'video+audio'` |  | 'video+audio' merges the best audio track into the chosen video quality. |
| `container` | `mp4` \| `mkv` \| `webm` | `'mp4'` |  | Video file type. |
| `audio_format` | `m4a` \| `mp3` \| `opus` \| `flac` \| `wav` | `'m4a'` |  | File type for audio_only. |
| `format_id` | string \| null | `null` |  | Exact id from video_probe, e.g. '136' or '136+140'. Overrides quality. |
| `output_dir` | string \| null | `null` |  | Folder to write into (under DVQA_ALLOWED_ROOTS). |
| `wait_seconds` | number | `30` | ≥0, ≤300 | Seconds to wait for completion before returning the job. |

Returns `Job`: A background download or ingest.

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Pass to video_job to poll, or video_cancel to stop. |
| `kind` | `ingest` \| `download` | What the job does. |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `interrupted` | `interrupted` means the server restarted mid-job; ingest again to resume. |
| `stage` | string | Current step: queued, download, transcribe, frames, embed or done. |
| `progress` | number | Overall completion from 0 to 1, weighted by how long each stage usually takes. |
| `message` | string | Latest human-readable progress line. |
| `source` | string | The URL or path being processed. |
| `video_id` | string \| null | Known once the source has been resolved; use it with every other tool. |
| `error` | string \| null | 'code: message — hint' when the job failed. |
| `result` | object \| null | When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the download report (path, codecs, size, whether audio was merged). |
| `next` | string \| null | The call to make to keep waiting, while the job is unfinished. |

## Index

Make videos searchable. Long work runs as jobs.

### `video_ingest`

**Index a video for questions** · writes · network · returns **`Job`**

Download (for a URL), transcribe, sample frames and embed, so the video can be searched and questioned. Speech and frame work run in parallel. Re-ingesting reuses finished stages, so a failed or interrupted job resumes. Returns a Job; its result holds the video_id, counts and timings.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `source` | string | **required** |  | A video URL or the absolute path of a local media file. |
| `quality` | string \| null | `null` |  | Download height for URLs; higher reads small text better. |
| `language` | string \| null | `null` |  | Spoken language code such as 'en'. Omit to detect. |
| `frame_interval` | number \| null | `null` | >0 | Seconds between frames; cuts are added. |
| `max_frames` | int \| null | `null` | >0 | Frames kept after de-duplication. |
| `transcribe` | bool | `True` |  | Transcribe the speech. |
| `frames` | bool | `True` |  | Sample and embed frames. |
| `force` | bool | `False` |  | Redo stages that already completed. |
| `wait_seconds` | number | `30` | ≥0, ≤300 | Seconds to wait for completion before returning the job. |

Returns `Job`: A background download or ingest.

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Pass to video_job to poll, or video_cancel to stop. |
| `kind` | `ingest` \| `download` | What the job does. |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `interrupted` | `interrupted` means the server restarted mid-job; ingest again to resume. |
| `stage` | string | Current step: queued, download, transcribe, frames, embed or done. |
| `progress` | number | Overall completion from 0 to 1, weighted by how long each stage usually takes. |
| `message` | string | Latest human-readable progress line. |
| `source` | string | The URL or path being processed. |
| `video_id` | string \| null | Known once the source has been resolved; use it with every other tool. |
| `error` | string \| null | 'code: message — hint' when the job failed. |
| `result` | object \| null | When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the download report (path, codecs, size, whether audio was merged). |
| `next` | string \| null | The call to make to keep waiting, while the job is unfinished. |

### `video_ingest_batch`

**Index several videos or a playlist** · writes · network · returns **`BatchResult`**

Start one ingest job per source. Playlist and channel URLs expand to their first `playlist_limit` videos. Jobs run DVQA_MAX_CONCURRENT_JOBS at a time and the rest queue; a bad source is reported in `skipped` without stopping the others.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `sources` | string[] | **required** | min 1, max 50 | URLs or local paths; playlists expand. |
| `playlist_limit` | int | `25` | ≥1, ≤200 | Videos taken from each playlist. |
| `quality` | string \| null | `null` |  | Download height for URLs. |
| `language` | string \| null | `null` |  | Spoken language code. Omit to detect. |
| `transcribe` | bool | `True` |  | Transcribe the speech. |
| `frames` | bool | `True` |  | Sample and embed frames. |
| `wait_seconds` | number | `0` | ≥0, ≤300 | Seconds to wait for completion before returning the job. |

Returns `BatchResult`:

| Field | Type | Description |
| --- | --- | --- |
| `jobs` | Job[] | One per video, in source order after playlist expansion and de-duplication. |
| `expanded` | string[] | Playlists that were expanded, as 'url -> N videos'. |
| `skipped` | string[] | Sources that were not started, as 'source: code: reason'. |

## Jobs

Follow and control background work.

### `video_job`

**Check a job** · read-only · local · returns **`Job`**

A job's status, stage and progress, optionally waiting for it to finish. Progress is also streamed as MCP progress notifications while waiting.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `job_id` | string | **required** |  | Job id from video_ingest, video_ingest_batch or video_download. |
| `wait_seconds` | number | `0` | ≥0, ≤300 | Seconds to wait for completion before returning the job. |

Returns `Job`: A background download or ingest.

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Pass to video_job to poll, or video_cancel to stop. |
| `kind` | `ingest` \| `download` | What the job does. |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `interrupted` | `interrupted` means the server restarted mid-job; ingest again to resume. |
| `stage` | string | Current step: queued, download, transcribe, frames, embed or done. |
| `progress` | number | Overall completion from 0 to 1, weighted by how long each stage usually takes. |
| `message` | string | Latest human-readable progress line. |
| `source` | string | The URL or path being processed. |
| `video_id` | string \| null | Known once the source has been resolved; use it with every other tool. |
| `error` | string \| null | 'code: message — hint' when the job failed. |
| `result` | object \| null | When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the download report (path, codecs, size, whether audio was merged). |
| `next` | string \| null | The call to make to keep waiting, while the job is unfinished. |

### `video_jobs`

**List jobs** · read-only · local · returns **`JobList`**

Recent download and ingest jobs, newest first.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `active_only` | bool | `False` |  | Only queued and running jobs. |
| `limit` | int | `20` | ≥1, ≤100 | Most jobs to return. |

Returns `JobList`:

| Field | Type | Description |
| --- | --- | --- |
| `jobs` | Job[] | Newest first. |

### `video_cancel`

**Cancel a job** · writes · local · returns **`CancelResult`**

Stop a queued or running job. Stages that already finished stay in the index and are reused.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `job_id` | string | **required** |  | Job id to stop. |

Returns `CancelResult`:

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Pass to video_job to poll, or video_cancel to stop. |
| `kind` | `ingest` \| `download` | What the job does. |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `interrupted` | `interrupted` means the server restarted mid-job; ingest again to resume. |
| `stage` | string | Current step: queued, download, transcribe, frames, embed or done. |
| `progress` | number | Overall completion from 0 to 1, weighted by how long each stage usually takes. |
| `message` | string | Latest human-readable progress line. |
| `source` | string | The URL or path being processed. |
| `video_id` | string \| null | Known once the source has been resolved; use it with every other tool. |
| `error` | string \| null | 'code: message — hint' when the job failed. |
| `result` | object \| null | When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the download report (path, codecs, size, whether audio was merged). |
| `next` | string \| null | The call to make to keep waiting, while the job is unfinished. |
| `cancel_requested` | bool | False when the job had already finished. |

## Retrieve

Find evidence across every indexed video.

### `video_ask`

**Gather evidence for a question** · read-only · local · returns **`AskResult`**

The moments most relevant to a question, as evidence: for each, the transcript around it with timestamps and the frames from it as images (in `moments` order). Answer from the evidence and cite times. With verify, each moment carries calibrated support probabilities and `evidence.supported` says whether the video answers the question at all.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `question` | string | **required** | min len 1 | The question to answer from the video(s). |
| `video_ids` | string[] \| null | `null` |  | Limit to these video ids. Omit for every indexed video. |
| `max_moments` | int | `5` | ≥1, ≤12 | Moments to return. |
| `frames_per_moment` | int | `2` | ≥0, ≤4 | Images attached per moment. |
| `image_px` | int | `768` | ≥256, ≤1568 | Long side of returned images, in pixels. |
| `verify` | bool \| null | `null` |  | Have Jev 1.14 judge whether each moment's transcript answers the question and whether each frame shows it, rerank by that, and rescue with a frames-only search when nothing is supported. Sends excerpts and frames to TypeSafe. Default: DVQA_VERIFY (off). |

Returns `AskResult`:

| Field | Type | Description |
| --- | --- | --- |
| `question` | string | The question asked. |
| `moments` | Moment[] | Most relevant first. |
| `evidence` | Evidence | How strongly anything matched. |
| `guidance` | string | How to use the evidence. |

### `video_search`

**Search inside videos** · read-only · local · returns **`SearchResult`**

Ranked hits from three channels fused by weighted reciprocal rank: meaning-based transcript search, keyword (BM25) search, and text-to-image search over frames. 'The slide with the pricing table' finds the frame even if nobody said it. Each hit lists the channels that matched it, with rank and raw score.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `query` | string | **required** | min len 1 | What to look for, in natural language. |
| `video_ids` | string[] \| null | `null` |  | Limit to these video ids. Omit for every indexed video. |
| `modalities` | (`transcript` \| `frames`)[] | `['transcript', 'frames']` |  | Search what was said, what was shown, or both. |
| `top_k` | int | `10` | ≥1, ≤50 | Hits to return. |
| `start` | number \| null | `null` | ≥0 | Only after this second. |
| `end` | number \| null | `null` | ≥0 | Only before this second. |

Returns `SearchResult`:

| Field | Type | Description |
| --- | --- | --- |
| `query` | string | The query, or the image path for image search. |
| `count` | int | Hits returned. |
| `hits` | Hit[] | Best first. |

### `video_search_by_image`

**Find frames that look like an image** · read-only · local · returns **`SearchResult`**

Where a picture, logo, chart or slide appears: frames that look like it, and passages about it.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `image_path` | string | **required** |  | Absolute path to a picture (jpg, png, webp). |
| `video_ids` | string[] \| null | `null` |  | Limit to these video ids. Omit for every indexed video. |
| `top_k` | int | `8` | ≥1, ≤50 | Hits to return. |

Returns `SearchResult`:

| Field | Type | Description |
| --- | --- | --- |
| `query` | string | The query, or the image path for image search. |
| `count` | int | Hits returned. |
| `hits` | Hit[] | Best first. |

### `video_find_phrase`

**Find every time a phrase is said** · read-only · local · returns **`PhraseResult`**

Every transcript segment containing the exact phrase, in time order. Use it to count mentions or collect every place a name or number comes up; use video_search for meaning rather than wording. A phrase split across two segments is not matched.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `phrase` | string | **required** | min len 2, max len 200 | Exact words, case-insensitive. |
| `video_ids` | string[] \| null | `null` |  | Limit to these video ids. Omit for every indexed video. |
| `limit` | int | `100` | ≥1, ≤500 | Most occurrences to return. |

Returns `PhraseResult`:

| Field | Type | Description |
| --- | --- | --- |
| `phrase` | string | What was searched for. |
| `count` | int | Hits returned. |
| `truncated` | bool | More occurrences exist beyond the limit. |
| `hits` | PhraseHit[] | In video then time order. |

## Inspect

Read, look at and cut one video.

### `video_transcript`

**Read the transcript** · read-only · local · returns **`TranscriptResult`**

What was said, for the whole video or a range: timestamped lines, plain text, SRT, WebVTT, or JSON segments. Page through long videos with start and end.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |
| `start` | number \| null | `null` | ≥0 | From this second. |
| `end` | number \| null | `null` | ≥0 | Up to this second. |
| `format` | `timestamped` \| `plain` \| `srt` \| `vtt` \| `json` | `'timestamped'` |  | Output layout. |
| `max_chars` | int | `60000` | ≥500, ≤400000 | Truncate beyond this. |

Returns `TranscriptResult`:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video. |
| `title` | string \| null | Its title. |
| `language` | string \| null | Spoken language. |
| `segment_count` | int | Segments in the requested range. |
| `format` | string | The layout returned. |
| `text` | string \| null | Rendered transcript, for timestamped, plain, srt and vtt. |
| `segments` | Segment[] \| null | Raw segments, for json. |
| `truncated` | bool | True when max_chars cut the output; page with start and end. |
| `note` | string \| null | Why the output is empty or truncated. |

### `video_frames`

**Look at frames** · read-only · local · returns **`FramesResult`**

Frames as images, at given timestamps or spread across a range. With exact=true each is decoded from the source at that instant and at higher resolution than the indexed samples, for reading small text.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |
| `timestamps` | number[] \| null | `null` | max 12 | Exact seconds, up to 12. |
| `start` | number \| null | `null` | ≥0 | Or: evenly spaced frames from this second... |
| `end` | number \| null | `null` | ≥0 | ...to this second. |
| `count` | int | `6` | ≥1, ≤12 | Frames across start..end. |
| `exact` | bool | `True` |  | Decode at the exact instant from the source file. |
| `image_px` | int | `1024` | ≥256, ≤1568 | Long side of returned images, in pixels. |

Returns `FramesResult`:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video. |
| `title` | string \| null | Its title. |
| `exact` | bool | True when frames were decoded from the source at the exact instant. |
| `frames` | FrameRef[] | Attached as images after the JSON in this order. |

### `video_chapters`

**Outline a video** · read-only · local · returns **`ChaptersResult`**

Chapters with spans, opening words and distinguishing keywords. Uses the uploader's chapters when the source has them; otherwise finds topic shifts in the transcript embeddings (frames for silent video).

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |
| `include_images` | bool | `False` |  | Attach one frame per chapter. |
| `image_px` | int | `512` | ≥256, ≤1024 | Long side of chapter images. |

Returns `ChaptersResult`:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video. |
| `title` | string \| null | Its title. |
| `chapters` | Chapter[] | In time order. |

### `video_clip`

**Export a clip** · writes · local · returns **`ClipResult`**

Cut a time range into its own MP4 with audio. Stream copy is instant; exact re-encodes.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |
| `start` | number | **required** | ≥0 | Clip start in seconds. |
| `end` | number | **required** | >0 | Clip end in seconds. |
| `exact` | bool | `False` |  | Re-encode for frame-accurate cuts; off snaps to keyframes. |
| `output_dir` | string \| null | `null` |  | Folder to write into (under DVQA_ALLOWED_ROOTS). |

Returns `ClipResult`:

| Field | Type | Description |
| --- | --- | --- |
| `path` | string | The MP4 written. |
| `duration` | number | Seconds. |
| `size_bytes` | int | File size. |
| `has_audio` | bool | Whether the clip has sound. |
| `exact` | bool | True when re-encoded for frame-accurate cuts. |

## Library

Manage what is indexed.

### `video_list`

**List indexed videos** · read-only · local · returns **`VideoList`**

Every video in the index with its id, title, length, status and index counts.

_No arguments._

Returns `VideoList`:

| Field | Type | Description |
| --- | --- | --- |
| `count` | int | Number of videos. |
| `videos` | VideoSummary[] | Newest first. |

### `video_info`

**Describe one video** · read-only · local · returns **`VideoDetail`**

Source, dimensions, language, index counts, completed stages and the uploader's description.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |

Returns `VideoDetail`:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Video id used by every other tool. |
| `title` | string \| null | Title from the site, or the file name. |
| `uploader` | string \| null | Channel or uploader, for URLs. |
| `source` | string | The URL or path it was ingested from. |
| `source_kind` | `url` \| `file` | Where it came from. |
| `webpage_url` | string \| null | Canonical page URL, for URLs. |
| `path` | string \| null | Media file the server reads from. |
| `duration` | number | Length in seconds. |
| `length` | string | Length as m:ss or h:mm:ss. |
| `width` | int | Frame width in pixels. |
| `height` | int | Frame height in pixels. |
| `fps` | number | Frame rate. |
| `has_audio` | bool | Whether the media has a sound track. |
| `language` | string \| null | Detected or given spoken language. |
| `status` | `pending` \| `processing` \| `ready` \| `failed` | Only `ready` videos are fully searchable. |
| `segment_count` | int \| null | Transcript segments (in listings). |
| `passage_count` | int \| null | Retrieval passages (in listings). |
| `frame_count` | int \| null | Indexed frames (in listings). |
| `counts` | IndexCounts | What is indexed for this video. |
| `stages_done` | string[] | Completed stages that a re-ingest will reuse. |
| `upload_date` | string \| null | YYYYMMDD, for URLs. |
| `description` | string \| null | Uploader's description, truncated to 2,000 characters. |

### `video_delete`

**Remove a video from the index** · writes · destructive · idempotent · local · returns **`DeleteResult`**

Delete a video's transcript, frames and vectors. A file you ingested by path is never deleted; only the server's own downloaded copy and extracted frames are.

| Argument | Type | Default | Limits | Description |
| --- | --- | --- | --- | --- |
| `video_id` | string | **required** |  | Video id from video_list or an ingest result. |
| `delete_files` | bool | `True` |  | Also delete the server's downloaded copy and frames. |

Returns `DeleteResult`:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video removed. |
| `removed` | bool | False when it was not in the index. |
| `files_deleted` | bool | Whether the server's own media copy and frames were deleted. |

### `video_status`

**Server status** · read-only · local · returns **`StatusReport`**

Readiness and capacity: FFmpeg, yt-dlp and its JavaScript runtimes, the transcription and embedding backends, limits, and index size. `ready` is false, with reasons in `problems`, when ingest would fail.

_No arguments._

Returns `StatusReport`:

| Field | Type | Description |
| --- | --- | --- |
| `version` | string | Server version. |
| `ready` | bool | True when ingest and search will work with the current backends. |
| `problems` | string[] | What is missing or degraded. |
| `ffmpeg` | string | Path, or 'missing'. |
| `ffprobe` | string | Path, or 'missing'. |
| `yt_dlp` | string | yt-dlp version. |
| `js_runtimes` | string[] | JavaScript runtimes yt-dlp can use for YouTube's challenge. |
| `transcription` | string | Active speech-to-text backend. |
| `embedding` | string | Active embedding model, dimension and where it runs. |
| `verification` | string | Evidence verifier (Jev) and whether video_ask uses it by default. |
| `state_dir` | string | Where the index and media live. |
| `limits` | Limits | Capacity settings. |
| `index` | IndexStats | Index size. |

## Nested types

### `Limits`

| Field | Type | Description |
| --- | --- | --- |
| `max_concurrent_jobs` | int | Jobs that run at once; more queue. |
| `cpu_workers` | int | Parallel FFmpeg decoders and image workers. |
| `max_duration_seconds` | int | Longest video accepted. |
| `max_frames_per_video` | int | Frames kept per video after de-duplication. |
| `embed_tokens_per_minute` | int | Embedding token budget. |

### `IndexStats`

| Field | Type | Description |
| --- | --- | --- |
| `videos` | int | Videos in the index. |
| `ready_videos` | int | Fully indexed videos. |
| `indexed_seconds` | number | Total duration of ready videos. |
| `segments` | int | Transcript segments. |
| `passages` | int | Retrieval passages. |
| `frames` | int | Indexed frames. |
| `vectors` | int | Stored embeddings. |
| `active_jobs` | int | Queued or running jobs. |
| `vector_sets` | object[] | Vector counts per model, dimension and kind. |
| `full_text_search` | bool | Whether SQLite FTS5 keyword search is available. |

### `Quality`

One entry in a downloader-style quality menu.

| Field | Type | Description |
| --- | --- | --- |
| `quality` | string | Height, such as '1080p'. |
| `label` | string | Menu label such as '720p MP4 · 56.23 MB'. |
| `has_sound_natively` | bool | This quality is served with its own audio. |
| `needs_audio_merge` | bool | This quality is served silent; video_download merges audio_format_id in. |
| `merged_size_bytes` | int \| null | Video plus merged audio, estimated from bitrate when the site does not say. |
| `video_format_id` | string | The video stream video_download would pick for this quality. |
| `audio_format_id` | string \| null | The audio stream that would be merged in, when one is needed. |

### `StreamFormat`

One raw stream the site offers.

| Field | Type | Description |
| --- | --- | --- |
| `format_id` | string | Pass to video_download as format_id, alone or joined as 'video+audio'. |
| `kind` | `video+audio` \| `video_only` \| `audio_only` | Which tracks the stream carries. |
| `height` | int \| null | Pixel height for video streams. |
| `fps` | number \| null | Frame rate for video streams. |
| `ext` | string \| null | Container extension. |
| `vcodec` | string | Video codec family, or 'none'. |
| `acodec` | string | Audio codec family, or 'none'. |
| `bitrate_kbps` | int \| null | Average total bitrate. |
| `size_bytes` | int \| null | Exact or approximate size; estimated from bitrate when unknown. |
| `language` | string \| null | Audio language, when the site labels it. |

### `Job`

A background download or ingest.

| Field | Type | Description |
| --- | --- | --- |
| `job_id` | string | Pass to video_job to poll, or video_cancel to stop. |
| `kind` | `ingest` \| `download` | What the job does. |
| `status` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `interrupted` | `interrupted` means the server restarted mid-job; ingest again to resume. |
| `stage` | string | Current step: queued, download, transcribe, frames, embed or done. |
| `progress` | number | Overall completion from 0 to 1, weighted by how long each stage usually takes. |
| `message` | string | Latest human-readable progress line. |
| `source` | string | The URL or path being processed. |
| `video_id` | string \| null | Known once the source has been resolved; use it with every other tool. |
| `error` | string \| null | 'code: message — hint' when the job failed. |
| `result` | object \| null | When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the download report (path, codecs, size, whether audio was merged). |
| `next` | string \| null | The call to make to keep waiting, while the job is unfinished. |

### `TranscriptLine`

| Field | Type | Description |
| --- | --- | --- |
| `at` | string | m:ss. |
| `start` | number | Seconds. |
| `text` | string | What was said. |

### `FrameRef`

| Field | Type | Description |
| --- | --- | --- |
| `at` | string | m:ss. |
| `t` | number | Seconds. |
| `path` | string \| null | Indexed frame file; null for frames decoded on the fly. |
| `shows` | number \| null | With verify: probability that the frame shows what the question asks about. |

### `Moment`

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video. |
| `title` | string \| null | Its title. |
| `start` | number | Window start in seconds. |
| `end` | number | Window end in seconds (at most 75 s after start). |
| `span` | string | Window as m:ss-m:ss. |
| `score` | number | Sum of fused scores of the hits merged into this window. |
| `matched_by` | string[] | Channels that contributed. |
| `transcript` | TranscriptLine[] | What was said in the window, with 8 s of context either side. |
| `frames` | FrameRef[] | Frames from the window, attached as images after the JSON in this order. |
| `transcript_answers` | number \| null | With verify: probability that the transcript window directly answers the question. |
| `support` | number \| null | With verify: the strongest of transcript_answers and the frames' `shows`; moments are ranked by it. |

### `Evidence`

Raw match strength, before rank fusion discards it. Compare values within one embedding model only.

| Field | Type | Description |
| --- | --- | --- |
| `best_said_similarity` | number \| null | Top transcript cosine similarity. |
| `best_shown_similarity` | number \| null | Top frame cosine similarity. |
| `keyword_matches` | int | Passages containing any of the question's words. |
| `verified` | bool | Whether the moments were checked by the verifier. |
| `verifier` | string \| null | Model that judged the evidence, such as 'jev-1.14.0'. |
| `supported` | bool \| null | With verify: true when some moment's support is at least 0.5. False means the video likely does not answer the question, or retrieval missed it. |
| `best_support` | number \| null | With verify: the highest moment support. |
| `rescued_by_frames` | bool | With verify: true when the supported moments came from a frames-only search after transcript evidence failed. |
| `verify_error` | string \| null | Set when verification was requested but failed; the moments are then unverified. |

### `ChannelMatch`

| Field | Type | Description |
| --- | --- | --- |
| `rank` | int | Position within that channel's ranking; 1 is best. |
| `score` | number | Cosine similarity for dense channels, BM25 for keyword. |

### `Hit`

| Field | Type | Description |
| --- | --- | --- |
| `kind` | `transcript` \| `frame` | A transcript passage or a frame. |
| `video_id` | string | The video. |
| `start` | number | Seconds. |
| `end` | number | Seconds; equal to start for frames. |
| `at` | string | Start as m:ss. |
| `score` | number | Fused reciprocal-rank score; only comparable within one result. |
| `matched_by` | map<string, ChannelMatch> | Channels that found it: said_semantic, said_keyword, shown (text queries); looks_like, talks_about (image queries). |
| `text` | string \| null | Passage text, for transcript hits. |
| `frame_path` | string \| null | Frame file, for frame hits. |

### `PhraseHit`

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | The video. |
| `title` | string \| null | Its title. |
| `start` | number | Segment start in seconds. |
| `end` | number | Segment end in seconds. |
| `at` | string | Start as m:ss. |
| `text` | string | The full segment containing the phrase. |

### `Segment`

| Field | Type | Description |
| --- | --- | --- |
| `start` | number | Seconds. |
| `end` | number | Seconds. |
| `text` | string | What was said. |

### `Chapter`

| Field | Type | Description |
| --- | --- | --- |
| `index` | int | 1-based. |
| `start` | number | Seconds. |
| `end` | number | Seconds. |
| `span` | string | m:ss-m:ss. |
| `title` | string \| null | Set when the uploader provided chapters. |
| `opens_with` | string | The first words spoken in the chapter. |
| `keywords` | string[] | Terms that distinguish this chapter from the rest of the video. |
| `frame_path` | string \| null | A frame from the chapter's opening seconds. |
| `source` | `uploader` \| `transcript` \| `frames` | Where the boundaries came from. |

### `VideoSummary`

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Video id used by every other tool. |
| `title` | string \| null | Title from the site, or the file name. |
| `uploader` | string \| null | Channel or uploader, for URLs. |
| `source` | string | The URL or path it was ingested from. |
| `source_kind` | `url` \| `file` | Where it came from. |
| `webpage_url` | string \| null | Canonical page URL, for URLs. |
| `path` | string \| null | Media file the server reads from. |
| `duration` | number | Length in seconds. |
| `length` | string | Length as m:ss or h:mm:ss. |
| `width` | int | Frame width in pixels. |
| `height` | int | Frame height in pixels. |
| `fps` | number | Frame rate. |
| `has_audio` | bool | Whether the media has a sound track. |
| `language` | string \| null | Detected or given spoken language. |
| `status` | `pending` \| `processing` \| `ready` \| `failed` | Only `ready` videos are fully searchable. |
| `segment_count` | int \| null | Transcript segments (in listings). |
| `passage_count` | int \| null | Retrieval passages (in listings). |
| `frame_count` | int \| null | Indexed frames (in listings). |

### `IndexCounts`

| Field | Type | Description |
| --- | --- | --- |
| `segments` | int | Transcript segments as Whisper produced them. |
| `passages` | int | Overlapping ~30-second passages used for retrieval. |
| `frames` | int | Distinct frames kept. |
| `vectors` | int | Embeddings stored (passages + frames). |

## Resources

| URI | Type | Description |
| --- | --- | --- |
| `video://library` | `application/json` | Every indexed video with id, title, length and status. |
| `video://{video_id}/transcript` | `text/plain` | The full transcript with [m:ss] timestamps. |
| `video://{video_id}/subtitles.srt` | `application/x-subrip` | The transcript as an SRT subtitle file. |
| `video://{video_id}/info` | `application/json` | Source, dimensions, language, index counts and description. |
| `video://{video_id}/frame/{seconds}` | `image/jpeg` | The indexed frame nearest to a time in seconds, as JPEG. |

## Prompts

| Prompt | Arguments | Description |
| --- | --- | --- |
| `video_brief` | `source` | Ingest a video if needed and produce a structured brief: thesis, chapters, key claims, open questions. |
| `video_fact_check` | `video_id`, `topic`? | List the checkable claims a video makes, each with its exact wording, timestamp and on-screen support. |
| `video_study_notes` | `video_id` | Turn a lecture or tutorial into structured notes with definitions, steps and the slides that matter. |
| `video_compare` | `video_ids`, `question` | Answer one question across several videos and show where they agree and disagree. |
| `video_find_moment` | `description` | Locate a moment by what was said or shown and verify it visually. |

## Error codes

| Code | Category | Retryable | What to do |
| --- | --- | --- | --- |
| `bad_url` | input | no | Pass a plain http(s) link to a single video. |
| `bad_quality` | input | no | Use 'best' or a height such as '720p'. |
| `bad_mode` | input | no | Use video+audio, video_only or audio_only. |
| `bad_container` | input | no | Use mp4, mkv or webm. |
| `bad_audio_format` | input | no | Use m4a, mp3, opus, flac or wav. |
| `bad_format` | input | no | Copy a format_id exactly from video_probe. |
| `bad_range` | input | no | Give start < end within the video's duration. |
| `nothing_to_do` | input | no | Enable at least one of transcribe or frames. |
| `playlist` | input | no | Use video_ingest_batch to ingest a playlist. |
| `too_many` | input | no | Split the request into smaller batches. |
| `private_host` | access | no | Set DVQA_ALLOW_PRIVATE_HOSTS=1 to permit private addresses. |
| `path_not_allowed` | access | no | Add the folder to DVQA_ALLOWED_ROOTS. |
| `not_found` | source | no | Check the id with video_list or video_jobs. |
| `not_a_file` | source | no | Pass the path of a media file, not a folder. |
| `not_media` | source | no | The file has no audio or video stream. |
| `source_missing` | source | no | Re-ingest the video; its downloaded copy was removed. |
| `too_long` | source | no | Raise DVQA_MAX_DURATION or pick a shorter video. |
| `probe_failed` | source | yes | The site refused or changed; try DVQA_COOKIES_FILE or retry later. |
| `download_failed` | source | yes | Retry; for sign-in walls set DVQA_COOKIES_FILE. |
| `download_skipped` | source | no | The video is live, too long or over DVQA_MAX_FILESIZE. |
| `no_frame` | source | no | Pick a timestamp inside the video. |
| `nothing_indexed` | source | no | The video had no speech and no frames to index. |
| `missing_tool` | backend | no | Install FFmpeg so ffmpeg and ffprobe are on PATH. |
| `ffmpeg_failed` | backend | no | FFmpeg could not read or write the media; check the file. |
| `no_transcriber` | backend | no | Install the mlx or faster-whisper extra, or set GROQ_API_KEY. |
| `transcribe_failed` | backend | yes | The transcription provider failed; retry the ingest. |
| `no_embed_key` | backend | no | Set JINA_API_KEY. |
| `embed_failed` | backend | yes | Retry; finished stages are reused. Lower DVQA_EMBED_TPM if rate-limited. |
| `config` | backend | no | Fix the named DVQA_* environment variable. |
| `verify_unavailable` | backend | no | Install the jev extra and set TYPESAFE_API_KEY, or call without verify. |
| `verify_failed` | backend | yes | Verification failed; the unverified evidence is still usable. |
| `cancelled` | state | no | The job was cancelled; run the ingest again to resume. |
