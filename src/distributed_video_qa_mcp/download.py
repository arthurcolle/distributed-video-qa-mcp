"""Link to file: list what a URL offers, then fetch it with audio merged in.

Most sites serve anything above 360p as a video-only stream and a separate audio
stream. `download` picks both and has FFmpeg mux them, which is the "combine
with audio" step that web downloaders make you do by hand.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import Cancelled, VideoQAError
from .netguard import check_url

log = logging.getLogger(__name__)

MODES = ("video+audio", "video_only", "audio_only")
CONTAINERS = ("mp4", "mkv", "webm")
AUDIO_FORMATS = ("m4a", "mp3", "opus", "flac", "wav")


class _YtdlpLogger:
    """Route yt-dlp output to logging; stdout belongs to the MCP stdio transport."""

    def debug(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        log.info("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        log.warning("yt-dlp: %s", message)


def js_runtimes() -> dict[str, dict[str, str]]:
    """JavaScript runtimes for YouTube's signature challenge, found explicitly.

    Without one yt-dlp still works but YouTube offers only low qualities. The
    `deno` extra installs Deno beside this interpreter, which may not be on PATH
    when an MCP host launches the server by absolute path.
    """
    found: dict[str, dict[str, str]] = {}
    beside = Path(sys.executable).parent / ("deno.exe" if os.name == "nt" else "deno")
    if beside.exists():
        found["deno"] = {"path": str(beside)}
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path and name not in found:
            found[name] = {"path": path}
    return found


def _base_options(settings: Settings) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _YtdlpLogger(),
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "restrictfilenames": True,
        "windowsfilenames": True,
        "ignore_no_formats_error": False,
        "color": "never",
    }
    runtimes = js_runtimes()
    if runtimes:
        options["js_runtimes"] = runtimes
    if settings.cookies_file:
        options["cookiefile"] = str(settings.cookies_file)
    return options


def _kind(fmt: dict[str, Any]) -> str | None:
    has_video = fmt.get("vcodec") not in (None, "none")
    has_audio = fmt.get("acodec") not in (None, "none")
    if has_video and has_audio:
        return "video+audio"
    if has_video:
        return "video_only"
    if has_audio:
        return "audio_only"
    return None


def _size(fmt: dict[str, Any], duration: float | None = None) -> int | None:
    size = fmt.get("filesize") or fmt.get("filesize_approx")
    if not size and fmt.get("tbr") and duration:
        size = float(fmt["tbr"]) * 1000 / 8 * float(duration)  # estimate from average bitrate
    return int(size) if size else None


def _human(size: int | None) -> str:
    if not size:
        return "size unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def summarize_formats(info: dict[str, Any]) -> dict[str, Any]:
    """Turn yt-dlp's format list into the menu a downloader page shows."""
    formats: list[dict[str, Any]] = []
    for fmt in info.get("formats") or []:
        kind = _kind(fmt)
        if kind is None or fmt.get("format_note") == "storyboard":
            continue
        formats.append(
            {
                "format_id": fmt.get("format_id"),
                "kind": kind,
                "height": fmt.get("height"),
                "fps": fmt.get("fps"),
                "ext": fmt.get("ext"),
                "vcodec": (fmt.get("vcodec") or "none").split(".")[0],
                "acodec": (fmt.get("acodec") or "none").split(".")[0],
                "bitrate_kbps": round(fmt["tbr"]) if fmt.get("tbr") else None,
                "size_bytes": _size(fmt, info.get("duration")),
                "language": fmt.get("language"),
            }
        )
    audio = [f for f in formats if f["kind"] == "audio_only"]
    best_audio = max(audio, key=lambda f: f["bitrate_kbps"] or 0, default=None)
    heights = sorted({f["height"] for f in formats if f["height"] and f["kind"] != "audio_only"})
    qualities: list[dict[str, Any]] = []
    for height in reversed(heights):
        at_height = [f for f in formats if f["height"] == height]
        muxed = [f for f in at_height if f["kind"] == "video+audio"]
        silent = [f for f in at_height if f["kind"] == "video_only"]
        pick = max(  # mirror the download preference: H.264 in MP4 first
            muxed or silent, key=lambda f: (f["vcodec"] in {"avc1", "h264"}, f["ext"] == "mp4", f["bitrate_kbps"] or 0)
        )
        needs_merge = not muxed
        total = pick["size_bytes"]
        if needs_merge and total and best_audio and best_audio["size_bytes"]:
            total += best_audio["size_bytes"]
        label = f"{height}p {str(pick['ext']).upper()} · {_human(total)}"
        qualities.append(
            {
                "quality": f"{height}p",
                "label": label,
                "has_sound_natively": not needs_merge,
                "needs_audio_merge": needs_merge,
                "merged_size_bytes": total,
                "video_format_id": pick["format_id"],
                "audio_format_id": best_audio["format_id"] if needs_merge and best_audio else None,
            }
        )
    return {
        "id": info.get("id"),
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url"),
        "thumbnail": info.get("thumbnail"),
        "is_live": bool(info.get("is_live")),
        "qualities": qualities,
        "audio_tracks": sorted(audio, key=lambda f: -(f["bitrate_kbps"] or 0))[:6],
        "formats": formats,
    }


def parse_quality(quality: str) -> int | None:
    """'best' -> None, '720p'/'720' -> 720."""
    text = quality.strip().lower()
    if text in {"best", "max", "highest", ""}:
        return None
    digits = text.removesuffix("p")
    if not digits.isdigit() or not 100 <= int(digits) <= 4320:
        raise VideoQAError("bad_quality", f"quality must be 'best' or a height such as '720p', got {quality!r}")
    return int(digits)


def format_selector(mode: str, height: int | None, container: str, format_id: str | None) -> tuple[str, list[str]]:
    """Build the yt-dlp format expression and the sort that decides ties."""
    if mode not in MODES:
        raise VideoQAError("bad_mode", f"mode must be one of {', '.join(MODES)}")
    # At equal resolution prefer H.264/AAC in MP4: it plays everywhere, whereas
    # AV1 and VP9 do not open in QuickTime or many editors.
    sort = ["res", "vcodec:h264", "acodec:aac", "ext:mp4:m4a"] if container == "mp4" else ["res"]
    if format_id:
        if not all(ch.isalnum() or ch in "+-_./" for ch in format_id):
            raise VideoQAError("bad_format", "format_id contains unexpected characters")
        # A bare video-only id gets the best audio added; yt-dlp ignores the
        # addition when the chosen format already has sound.
        if mode == "video+audio" and "+" not in format_id:
            return f"{format_id}+bestaudio/{format_id}", sort
        return format_id, sort
    cap = f"[height<={height}]" if height else ""
    if mode == "video+audio":
        return f"bestvideo*{cap}+bestaudio/best{cap}/bestvideo*+bestaudio/best", sort
    if mode == "video_only":
        return f"bestvideo{cap}/bestvideo", sort
    return "bestaudio/best", sort


async def probe(url: str, settings: Settings) -> dict[str, Any]:
    """Fetch metadata and the available formats without downloading anything."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    url = check_url(url, allow_private=settings.allow_private_hosts)

    def work() -> dict[str, Any]:
        with yt_dlp.YoutubeDL(_base_options(settings)) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except DownloadError as exc:
                raise VideoQAError("probe_failed", _clean(str(exc))) from exc
            return ydl.sanitize_info(info)  # type: ignore[return-value]

    info = await asyncio.to_thread(work)
    if info.get("_type") == "playlist":
        raise VideoQAError("playlist", "the URL is a playlist; pass a single video URL")
    return info


def _clean(message: str) -> str:
    return message.replace("ERROR: ", "").strip()[:600]


@dataclass(frozen=True)
class Download:
    path: Path
    info: dict[str, Any]
    mode: str
    height: int | None
    merged: bool


async def download(
    url: str,
    dest_dir: Path,
    settings: Settings,
    *,
    quality: str = "best",
    mode: str = "video+audio",
    container: str = "mp4",
    audio_format: str = "m4a",
    format_id: str | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> Download:
    """Download one video. With mode 'video+audio' the result always has sound when the source has any."""
    import yt_dlp
    from yt_dlp.utils import DownloadError, match_filter_func

    url = check_url(url, allow_private=settings.allow_private_hosts)
    if container not in CONTAINERS:
        raise VideoQAError("bad_container", f"container must be one of {', '.join(CONTAINERS)}")
    if audio_format not in AUDIO_FORMATS:
        raise VideoQAError("bad_audio_format", f"audio_format must be one of {', '.join(AUDIO_FORMATS)}")
    height = parse_quality(quality)
    selector, sort = format_selector(mode, height, container, format_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    done: dict[str, int] = {}

    def hook(event: dict[str, Any]) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        name = str(event.get("filename") or "")
        if event.get("status") == "downloading":
            totals[name] = int(event.get("total_bytes") or event.get("total_bytes_estimate") or 0)
            done[name] = int(event.get("downloaded_bytes") or 0)
        elif event.get("status") == "finished":
            done[name] = totals[name] = int(event.get("total_bytes") or done.get(name, 0) or 1)
        if on_progress and sum(totals.values()) > 0:
            fraction = min(1.0, sum(done.values()) / sum(totals.values()))
            on_progress(fraction, f"downloaded {_human(sum(done.values()))}")

    options = _base_options(settings)
    options.update(
        {
            "format": selector,
            "format_sort": sort,
            "outtmpl": str(dest_dir / "%(extractor)s-%(id)s-%(format_id)s.%(ext)s"),
            "concurrent_fragment_downloads": settings.download_fragments,
            "continuedl": True,
            "overwrites": False,
            "max_filesize": settings.max_filesize,
            "progress_hooks": [hook],
            "match_filter": match_filter_func(f"!is_live & duration <=? {settings.max_duration}"),
        }
    )
    if mode == "audio_only":
        options["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": audio_format}]
    else:
        options["merge_output_format"] = container

    def work() -> dict[str, Any]:
        with yt_dlp.YoutubeDL(options) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except DownloadError as exc:
                if cancel is not None and cancel.is_set():
                    raise Cancelled() from exc
                raise VideoQAError("download_failed", _clean(str(exc))) from exc
            return ydl.sanitize_info(info)  # type: ignore[return-value]

    info = await asyncio.to_thread(work)
    if info.get("_type") == "playlist":
        raise VideoQAError("playlist", "the URL is a playlist; pass a single video URL")
    requested = info.get("requested_downloads") or []
    if not requested or not requested[0].get("filepath"):
        raise VideoQAError(
            "download_skipped",
            "nothing was downloaded; the video is live, too long, or larger than DVQA_MAX_FILESIZE",
        )
    path = Path(requested[0]["filepath"])
    if not path.exists():
        raise VideoQAError("download_failed", "the downloader reported success but wrote no file")
    parts = requested[0].get("requested_formats") or []
    return Download(path=path, info=info, mode=mode, height=info.get("height"), merged=len(parts) > 1)
