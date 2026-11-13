"""Typed tool results.

Each tool declares one of these as its output. The SDK publishes the JSON
Schema as the tool's `outputSchema`, validates every result against it, and
sends it as `structuredContent`. Tools that also return images send the same
structure first, then the image blocks.

Field docstrings are the field descriptions in the published schema and in
docs/tool-contract.md, so every field must have one (a test enforces it).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


# ---------------------------------------------------------------- jobs


class Job(_Model):
    """A background download or ingest."""

    job_id: str
    """Pass to video_job to poll, or video_cancel to stop."""
    kind: Literal["ingest", "download"]
    """What the job does."""
    status: JobStatus
    """`interrupted` means the server restarted mid-job; ingest again to resume."""
    stage: str
    """Current step: queued, download, transcribe, frames, embed or done."""
    progress: float
    """Overall completion from 0 to 1, weighted by how long each stage usually takes."""
    message: str
    """Latest human-readable progress line."""
    source: str
    """The URL or path being processed."""
    video_id: str | None = None
    """Known once the source has been resolved; use it with every other tool."""
    error: str | None = None
    """'code: message — hint' when the job failed."""
    result: dict | None = None
    """When succeeded: the ingest report (video_id, counts, stages, embedding tokens, elapsed seconds) or the
    download report (path, codecs, size, whether audio was merged)."""
    next: str | None = None
    """The call to make to keep waiting, while the job is unfinished."""


class JobList(_Model):
    jobs: list[Job]
    """Newest first."""


class BatchResult(_Model):
    jobs: list[Job]
    """One per video, in source order after playlist expansion and de-duplication."""
    expanded: list[str] = []
    """Playlists that were expanded, as 'url -> N videos'."""
    skipped: list[str] = []
    """Sources that were not started, as 'source: code: reason'."""


class CancelResult(Job):
    cancel_requested: bool
    """False when the job had already finished."""


# ---------------------------------------------------------------- acquisition


class Quality(_Model):
    """One entry in a downloader-style quality menu."""

    quality: str
    """Height, such as '1080p'."""
    label: str
    """Menu label such as '720p MP4 · 56.23 MB'."""
    has_sound_natively: bool
    """This quality is served with its own audio."""
    needs_audio_merge: bool
    """This quality is served silent; video_download merges audio_format_id in."""
    merged_size_bytes: int | None = None
    """Video plus merged audio, estimated from bitrate when the site does not say."""
    video_format_id: str
    """The video stream video_download would pick for this quality."""
    audio_format_id: str | None = None
    """The audio stream that would be merged in, when one is needed."""


class StreamFormat(_Model):
    """One raw stream the site offers."""

    format_id: str
    """Pass to video_download as format_id, alone or joined as 'video+audio'."""
    kind: Literal["video+audio", "video_only", "audio_only"]
    """Which tracks the stream carries."""
    height: int | None = None
    """Pixel height for video streams."""
    fps: float | None = None
    """Frame rate for video streams."""
    ext: str | None = None
    """Container extension."""
    vcodec: str
    """Video codec family, or 'none'."""
    acodec: str
    """Audio codec family, or 'none'."""
    bitrate_kbps: int | None = None
    """Average total bitrate."""
    size_bytes: int | None = None
    """Exact or approximate size; estimated from bitrate when unknown."""
    language: str | None = None
    """Audio language, when the site labels it."""


class ProbeResult(_Model):
    id: str | None
    """The site's id for the video."""
    extractor: str | None
    """yt-dlp extractor that handled the URL, such as 'Youtube'."""
    title: str | None
    """Video title."""
    uploader: str | None
    """Channel or uploader name."""
    upload_date: str | None
    """YYYYMMDD."""
    duration: float | None
    """Length in seconds."""
    webpage_url: str | None
    """Canonical page URL."""
    thumbnail: str | None
    """Thumbnail image URL."""
    is_live: bool
    """Live streams cannot be ingested."""
    qualities: list[Quality]
    """Best first, one entry per available height."""
    audio_tracks: list[StreamFormat]
    """Best audio streams first."""
    formats: list[StreamFormat] | None = None
    """Every raw stream, when all_formats is set."""


# ---------------------------------------------------------------- library


class VideoSummary(_Model):
    id: str
    """Video id used by every other tool."""
    title: str | None
    """Title from the site, or the file name."""
    uploader: str | None = None
    """Channel or uploader, for URLs."""
    source: str
    """The URL or path it was ingested from."""
    source_kind: Literal["url", "file"]
    """Where it came from."""
    webpage_url: str | None = None
    """Canonical page URL, for URLs."""
    path: str | None = None
    """Media file the server reads from."""
    duration: float
    """Length in seconds."""
    length: str
    """Length as m:ss or h:mm:ss."""
    width: int
    """Frame width in pixels."""
    height: int
    """Frame height in pixels."""
    fps: float
    """Frame rate."""
    has_audio: bool
    """Whether the media has a sound track."""
    language: str | None = None
    """Detected or given spoken language."""
    status: Literal["pending", "processing", "ready", "failed"]
    """Only `ready` videos are fully searchable."""
    segment_count: int | None = None
    """Transcript segments (in listings)."""
    passage_count: int | None = None
    """Retrieval passages (in listings)."""
    frame_count: int | None = None
    """Indexed frames (in listings)."""


class VideoList(_Model):
    count: int
    """Number of videos."""
    videos: list[VideoSummary]
    """Newest first."""


class IndexCounts(_Model):
    segments: int
    """Transcript segments as Whisper produced them."""
    passages: int
    """Overlapping ~30-second passages used for retrieval."""
    frames: int
    """Distinct frames kept."""
    vectors: int
    """Embeddings stored (passages + frames)."""


class VideoDetail(VideoSummary):
    counts: IndexCounts
    """What is indexed for this video."""
    stages_done: list[str]
    """Completed stages that a re-ingest will reuse."""
    upload_date: str | None = None
    """YYYYMMDD, for URLs."""
    description: str | None = None
    """Uploader's description, truncated to 2,000 characters."""


class DeleteResult(_Model):
    video_id: str
    """The video removed."""
    removed: bool
    """False when it was not in the index."""
    files_deleted: bool
    """Whether the server's own media copy and frames were deleted."""


# ---------------------------------------------------------------- reading


class Segment(_Model):
    start: float
    """Seconds."""
    end: float
    """Seconds."""
    text: str
    """What was said."""


class TranscriptResult(_Model):
    video_id: str
    """The video."""
    title: str | None
    """Its title."""
    language: str | None
    """Spoken language."""
    segment_count: int
    """Segments in the requested range."""
    format: str
    """The layout returned."""
    text: str | None = None
    """Rendered transcript, for timestamped, plain, srt and vtt."""
    segments: list[Segment] | None = None
    """Raw segments, for json."""
    truncated: bool
    """True when max_chars cut the output; page with start and end."""
    note: str | None = None
    """Why the output is empty or truncated."""


class PhraseHit(_Model):
    video_id: str
    """The video."""
    title: str | None
    """Its title."""
    start: float
    """Segment start in seconds."""
    end: float
    """Segment end in seconds."""
    at: str
    """Start as m:ss."""
    text: str
    """The full segment containing the phrase."""


class PhraseResult(_Model):
    phrase: str
    """What was searched for."""
    count: int
    """Hits returned."""
    truncated: bool
    """More occurrences exist beyond the limit."""
    hits: list[PhraseHit]
    """In video then time order."""


# ---------------------------------------------------------------- retrieval


class ChannelMatch(_Model):
    rank: int
    """Position within that channel's ranking; 1 is best."""
    score: float
    """Cosine similarity for dense channels, BM25 for keyword."""


class Hit(_Model):
    kind: Literal["transcript", "frame"]
    """A transcript passage or a frame."""
    video_id: str
    """The video."""
    start: float
    """Seconds."""
    end: float
    """Seconds; equal to start for frames."""
    at: str
    """Start as m:ss."""
    score: float
    """Fused reciprocal-rank score; only comparable within one result."""
    matched_by: dict[str, ChannelMatch]
    """Channels that found it: said_semantic, said_keyword, shown (text queries); looks_like, talks_about
    (image queries)."""
    text: str | None = None
    """Passage text, for transcript hits."""
    frame_path: str | None = None
    """Frame file, for frame hits."""


class SearchResult(_Model):
    query: str
    """The query, or the image path for image search."""
    count: int
    """Hits returned."""
    hits: list[Hit]
    """Best first."""


class TranscriptLine(_Model):
    at: str
    """m:ss."""
    start: float
    """Seconds."""
    text: str
    """What was said."""


class FrameRef(_Model):
    at: str
    """m:ss."""
    t: float
    """Seconds."""
    path: str | None = None
    """Indexed frame file; null for frames decoded on the fly."""


class Moment(_Model):
    video_id: str
    """The video."""
    title: str | None
    """Its title."""
    start: float
    """Window start in seconds."""
    end: float
    """Window end in seconds (at most 75 s after start)."""
    span: str
    """Window as m:ss-m:ss."""
    score: float
    """Sum of fused scores of the hits merged into this window."""
    matched_by: list[str]
    """Channels that contributed."""
    transcript: list[TranscriptLine]
    """What was said in the window, with 8 s of context either side."""
    frames: list[FrameRef]
    """Frames from the window, attached as images after the JSON in this order."""


class Evidence(_Model):
    """Raw match strength, before rank fusion discards it. Compare values within one embedding model only."""

    best_said_similarity: float | None = None
    """Top transcript cosine similarity."""
    best_shown_similarity: float | None = None
    """Top frame cosine similarity."""
    keyword_matches: int
    """Passages containing any of the question's words."""


class AskResult(_Model):
    question: str
    """The question asked."""
    moments: list[Moment]
    """Most relevant first."""
    evidence: Evidence
    """How strongly anything matched."""
    guidance: str
    """How to use the evidence."""


class FramesResult(_Model):
    video_id: str
    """The video."""
    title: str | None
    """Its title."""
    exact: bool
    """True when frames were decoded from the source at the exact instant."""
    frames: list[FrameRef]
    """Attached as images after the JSON in this order."""


class Chapter(_Model):
    index: int
    """1-based."""
    start: float
    """Seconds."""
    end: float
    """Seconds."""
    span: str
    """m:ss-m:ss."""
    title: str | None = None
    """Set when the uploader provided chapters."""
    opens_with: str
    """The first words spoken in the chapter."""
    keywords: list[str]
    """Terms that distinguish this chapter from the rest of the video."""
    frame_path: str | None
    """A frame from the chapter's opening seconds."""
    source: Literal["uploader", "transcript", "frames"]
    """Where the boundaries came from."""


class ChaptersResult(_Model):
    video_id: str
    """The video."""
    title: str | None
    """Its title."""
    chapters: list[Chapter]
    """In time order."""


class ClipResult(_Model):
    path: str
    """The MP4 written."""
    duration: float
    """Seconds."""
    size_bytes: int
    """File size."""
    has_audio: bool
    """Whether the clip has sound."""
    exact: bool
    """True when re-encoded for frame-accurate cuts."""


# ---------------------------------------------------------------- status


class Limits(_Model):
    max_concurrent_jobs: int
    """Jobs that run at once; more queue."""
    cpu_workers: int
    """Parallel FFmpeg decoders and image workers."""
    max_duration_seconds: int
    """Longest video accepted."""
    max_frames_per_video: int
    """Frames kept per video after de-duplication."""
    embed_tokens_per_minute: int
    """Embedding token budget."""


class IndexStats(_Model):
    videos: int
    """Videos in the index."""
    ready_videos: int
    """Fully indexed videos."""
    indexed_seconds: float
    """Total duration of ready videos."""
    segments: int
    """Transcript segments."""
    passages: int
    """Retrieval passages."""
    frames: int
    """Indexed frames."""
    vectors: int
    """Stored embeddings."""
    active_jobs: int
    """Queued or running jobs."""
    vector_sets: list[dict]
    """Vector counts per model, dimension and kind."""
    full_text_search: bool
    """Whether SQLite FTS5 keyword search is available."""


class StatusReport(_Model):
    version: str
    """Server version."""
    ready: bool
    """True when ingest and search will work with the current backends."""
    problems: list[str]
    """What is missing or degraded."""
    ffmpeg: str
    """Path, or 'missing'."""
    ffprobe: str
    """Path, or 'missing'."""
    yt_dlp: str
    """yt-dlp version."""
    js_runtimes: list[str]
    """JavaScript runtimes yt-dlp can use for YouTube's challenge."""
    transcription: str
    """Active speech-to-text backend."""
    embedding: str
    """Active embedding model and dimension."""
    state_dir: str
    """Where the index and media live."""
    limits: Limits
    """Capacity settings."""
    index: IndexStats
    """Index size."""
