"""State shared by every tool, resource and prompt, and the conversions to typed results."""

from __future__ import annotations

import asyncio
import io
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, Image
from mcp.types import TextContent
from PIL import Image as PILImage

from . import __version__, retrieval
from . import download as dl
from . import transcribe as stt
from .config import Settings
from .embed import Embedder, make_embedder
from .errors import VideoQAError
from .models import Job, StatusReport, VideoDetail, VideoSummary
from .pipeline import TERMINAL, Pipeline
from .store import Store


def jpeg(source: Path | bytes, max_px: int) -> bytes:
    """Re-encode an image as JPEG with its long side capped at `max_px`."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    with PILImage.open(handle) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px), PILImage.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def _srt_time(seconds: float, sep: str) -> str:
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def render_transcript(segments: Sequence[dict[str, Any]], fmt: str) -> str:
    if fmt == "srt":
        return "\n".join(
            f"{i}\n{_srt_time(s['start'], ',')} --> {_srt_time(s['end'], ',')}\n{s['text']}\n"
            for i, s in enumerate(segments, start=1)
        )
    if fmt == "vtt":
        body = "\n".join(
            f"{_srt_time(s['start'], '.')} --> {_srt_time(s['end'], '.')}\n{s['text']}\n" for s in segments
        )
        return "WEBVTT\n\n" + body
    if fmt == "plain":
        return " ".join(s["text"] for s in segments)
    return "\n".join(f"[{retrieval.clock(s['start'])}] {s['text']}" for s in segments)


def job_model(job: dict[str, Any]) -> Job:
    unfinished = job["status"] not in TERMINAL
    return Job(
        job_id=job["id"],
        kind=job["kind"],
        status=job["status"],
        stage=job["stage"] or "queued",
        progress=job["progress"],
        message=job["message"],
        source=job["source"],
        video_id=job["video_id"],
        error=job.get("error"),
        result=job.get("result"),
        next=f"video_job(job_id='{job['id']}', wait_seconds=60)" if unfinished else None,
    )


_SUMMARY_FIELDS = set(VideoSummary.model_fields) - {"length"}


def video_summary(video: dict[str, Any]) -> VideoSummary:
    fields = {k: v for k, v in video.items() if k in _SUMMARY_FIELDS}
    return VideoSummary(**fields, length=retrieval.clock(video.get("duration") or 0))


class App:
    """Everything the tools share. Built once per process."""

    def __init__(self, settings: Settings) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.store = Store(settings.db_path)
        self.store.interrupt_stale_jobs()
        self.pipeline = Pipeline(settings, self.store, lambda: make_embedder(settings))

    @property
    def embedder(self) -> Embedder:
        return self.pipeline.embedder

    # ------------------------------------------------------------ lookups

    def video(self, video_id: str) -> dict[str, Any]:
        video = self.store.get_video(video_id.strip())
        if video is None:
            known = ", ".join(v["id"] for v in self.store.list_videos()[:8]) or "none indexed yet"
            raise VideoQAError("not_found", f"no video {video_id!r}; known ids: {known}")
        return video

    def video_detail(self, video_id: str) -> VideoDetail:
        video = self.video(video_id)
        summary = video_summary(video).model_dump()
        return VideoDetail(
            **summary,
            counts=self.store.counts(video["id"]),
            stages_done=sorted(video["stages"]),
            upload_date=video["meta"].get("upload_date"),
            description=video["meta"].get("description"),
        )

    def scope(self, video_ids: Sequence[str] | None) -> list[str] | None:
        if video_ids is None:
            return None
        return [self.video(video_id)["id"] for video_id in video_ids]

    def title(self, video_id: str) -> str | None:
        return (self.store.get_video(video_id) or {}).get("title")

    def source_file(self, video: dict[str, Any]) -> Path:
        path = Path(video.get("path") or "")
        if not path.is_file():
            raise VideoQAError("source_missing", f"the source file for {video['id']} is gone: {path}")
        return path

    def output_dir(self, requested: str | None, default: Path) -> Path:
        path = default if not requested else self.settings.check_path(requested, must_exist=False)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------ jobs

    async def wait(self, job_id: str, seconds: float, ctx: Context | None = None) -> Job:
        async def tick(job: dict[str, Any]) -> None:
            if ctx is not None:
                await ctx.report_progress(float(job["progress"]), 1.0, job["message"])

        return job_model(await self.pipeline.wait(job_id, seconds, tick))

    async def wait_all(self, job_ids: Sequence[str], seconds: float, ctx: Context | None = None) -> list[Job]:
        """Wait on several jobs under one deadline, reporting the mean progress."""

        async def tick(_: dict[str, Any]) -> None:
            if ctx is None:
                return
            jobs = [self.store.get_job(job_id) or {} for job_id in job_ids]
            done = sum(float(job.get("progress") or 0.0) for job in jobs) / max(len(jobs), 1)
            finished = sum(job.get("status") in TERMINAL for job in jobs)
            await ctx.report_progress(done, 1.0, f"{finished}/{len(jobs)} jobs finished")

        results = await asyncio.gather(
            *(self.pipeline.wait(job_id, seconds, tick if i == 0 else None) for i, job_id in enumerate(job_ids))
        )
        return [job_model(job) for job in results]

    # ------------------------------------------------------------ images

    async def images(self, items: Sequence[tuple[str, Path | bytes]], max_px: int) -> list[Any]:
        """Caption + image pairs as content blocks. A missing file becomes a text note."""
        blocks: list[Any] = []
        for label, source in items:
            try:
                data = await asyncio.to_thread(jpeg, source, max_px)
            except OSError:
                blocks.append(TextContent(type="text", text=f"{label}: image file is missing"))
                continue
            blocks.append(TextContent(type="text", text=label))
            blocks.append(Image(data=data, format="jpeg").to_image_content())
        return blocks

    # ------------------------------------------------------------ status

    def status(self) -> StatusReport:
        settings = self.settings
        tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
        try:
            import yt_dlp.version

            ytdlp = yt_dlp.version.__version__
        except Exception as exc:  # noqa: BLE001
            ytdlp = f"unavailable ({exc})"
        if settings.embed_backend == "jina" and not settings.embed_api_key:
            embedding = "unavailable (set JINA_API_KEY)"
        else:
            model = settings.embed_model if settings.embed_backend == "jina" else "hash-test-embedder"
            embedding = f"{model} at {settings.embed_dim} dimensions"
        transcription = stt.describe_backend(settings)
        runtimes = sorted(dl.js_runtimes())
        problems = [f"{name} is not on PATH" for name, path in tools.items() if not path]
        if embedding.startswith("unavailable"):
            problems.append("no embedding backend: set JINA_API_KEY")
        if transcription.startswith("unavailable"):
            problems.append("no transcription backend; ingest will fail unless transcribe=false")
        if not runtimes:
            problems.append("no JavaScript runtime; YouTube will offer only low qualities")
        blocking = [p for p in problems if "JavaScript" not in p]
        return StatusReport(
            version=__version__,
            ready=not blocking,
            problems=problems,
            ffmpeg=tools["ffmpeg"] or "missing",
            ffprobe=tools["ffprobe"] or "missing",
            yt_dlp=ytdlp,
            js_runtimes=runtimes,
            transcription=transcription,
            embedding=embedding,
            state_dir=str(settings.state_dir),
            limits={
                "max_concurrent_jobs": settings.max_concurrent_jobs,
                "cpu_workers": settings.cpu_workers,
                "max_duration_seconds": settings.max_duration,
                "max_frames_per_video": settings.max_frames,
                "embed_tokens_per_minute": settings.embed_tokens_per_minute,
            },
            index=self.store.stats(),
        )

    async def shutdown(self) -> None:
        await self.pipeline.shutdown()
        self.store.close()
