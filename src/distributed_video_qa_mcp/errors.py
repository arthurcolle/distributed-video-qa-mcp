"""Stable error codes.

Every expected failure carries a code from `CODES`. A failed tool call returns
`is_error: true` with the text

    Error executing tool <tool>: <code>: <message> — <hint>[ Retryable.]

so callers and models can branch on the code without parsing prose. The registry also feeds the generated tool
contract, and a test fails if code anywhere raises an unregistered code.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp.server.mcpserver.exceptions import ToolError


@dataclass(frozen=True)
class ErrorSpec:
    category: str  # input | access | source | backend | state
    retryable: bool
    hint: str


CODES: dict[str, ErrorSpec] = {
    # input: the request itself is wrong; fix the arguments
    "bad_url": ErrorSpec("input", False, "Pass a plain http(s) link to a single video."),
    "bad_quality": ErrorSpec("input", False, "Use 'best' or a height such as '720p'."),
    "bad_mode": ErrorSpec("input", False, "Use video+audio, video_only or audio_only."),
    "bad_container": ErrorSpec("input", False, "Use mp4, mkv or webm."),
    "bad_audio_format": ErrorSpec("input", False, "Use m4a, mp3, opus, flac or wav."),
    "bad_format": ErrorSpec("input", False, "Copy a format_id exactly from video_probe."),
    "bad_range": ErrorSpec("input", False, "Give start < end within the video's duration."),
    "nothing_to_do": ErrorSpec("input", False, "Enable at least one of transcribe or frames."),
    "playlist": ErrorSpec("input", False, "Use video_ingest_batch to ingest a playlist."),
    "too_many": ErrorSpec("input", False, "Split the request into smaller batches."),
    # access: the server refuses on policy
    "private_host": ErrorSpec("access", False, "Set DVQA_ALLOW_PRIVATE_HOSTS=1 to permit private addresses."),
    "path_not_allowed": ErrorSpec("access", False, "Add the folder to DVQA_ALLOWED_ROOTS."),
    # source: the media or the site is the problem
    "not_found": ErrorSpec("source", False, "Check the id with video_list or video_jobs."),
    "not_a_file": ErrorSpec("source", False, "Pass the path of a media file, not a folder."),
    "not_media": ErrorSpec("source", False, "The file has no audio or video stream."),
    "source_missing": ErrorSpec("source", False, "Re-ingest the video; its downloaded copy was removed."),
    "too_long": ErrorSpec("source", False, "Raise DVQA_MAX_DURATION or pick a shorter video."),
    "probe_failed": ErrorSpec("source", True, "The site refused or changed; try DVQA_COOKIES_FILE or retry later."),
    "download_failed": ErrorSpec("source", True, "Retry; for sign-in walls set DVQA_COOKIES_FILE."),
    "download_skipped": ErrorSpec("source", False, "The video is live, too long or over DVQA_MAX_FILESIZE."),
    "no_frame": ErrorSpec("source", False, "Pick a timestamp inside the video."),
    "nothing_indexed": ErrorSpec("source", False, "The video had no speech and no frames to index."),
    # backend: a tool or provider the server depends on
    "missing_tool": ErrorSpec("backend", False, "Install FFmpeg so ffmpeg and ffprobe are on PATH."),
    "ffmpeg_failed": ErrorSpec("backend", False, "FFmpeg could not read or write the media; check the file."),
    "no_transcriber": ErrorSpec("backend", False, "Install the mlx or faster-whisper extra, or set GROQ_API_KEY."),
    "transcribe_failed": ErrorSpec("backend", True, "The transcription provider failed; retry the ingest."),
    "no_embed_key": ErrorSpec("backend", False, "Set JINA_API_KEY."),
    "embed_failed": ErrorSpec(
        "backend", True, "Retry; finished stages are reused. Lower DVQA_EMBED_TPM if rate-limited."
    ),
    "config": ErrorSpec("backend", False, "Fix the named DVQA_* environment variable."),
    # state
    "cancelled": ErrorSpec("state", False, "The job was cancelled; run the ingest again to resume."),
}


class VideoQAError(ToolError):
    """An expected failure with a message that is safe to show to the MCP client.

    Subclassing ToolError makes the SDK return the message to the model as an
    error result; any other exception is reported only as a generic crash.
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in CODES:
            raise ValueError(f"unregistered error code {code!r}")
        spec = CODES[code]
        retry = " Retryable." if spec.retryable else ""
        super().__init__(f"{code}: {message} — {spec.hint}{retry}")
        self.code = code
        self.message = message
        self.spec = spec


class Cancelled(VideoQAError):
    def __init__(self, message: str = "job was cancelled") -> None:
        super().__init__("cancelled", message)
