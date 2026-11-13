"""The MCP tool surface.

The server retrieves evidence; the model that calls it does the reasoning. A
question comes back as timestamped transcript excerpts plus the actual frames
as images, so the caller reads the screen itself instead of trusting a caption.
"""

from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.types import ToolAnnotations
from PIL import Image as PILImage
from pydantic import Field

from . import __version__, media, retrieval
from . import download as dl
from . import transcribe as stt
from .config import Settings
from .embed import make_embedder
from .errors import VideoQAError
from .netguard import check_url, is_url
from .pipeline import TERMINAL, IngestOptions, Pipeline
from .store import Store

INSTRUCTIONS = """\
Download, index and question video.

Typical flow:
1. video_ingest(source) with a URL or a local file path. It returns a job; call video_job(job_id, wait_seconds=60) \
until status is "succeeded". The result carries the video_id.
2. video_ask(question) to get the most relevant moments: transcript excerpts with timestamps and the frames \
themselves as images. Answer from that evidence and cite timestamps.
3. Use video_search for a ranked hit list, video_transcript for exact wording, video_frames to look at any \
moment, video_chapters for structure, video_clip to export a segment.

To only save a file, use video_probe(url) to see the available qualities and video_download(url). Qualities \
above 360p are usually served without sound; video_download merges the best audio track in by default.
"""

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
READ_WEB = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITE_WEB = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTROY = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)

VideoIds = Annotated[
    list[str] | None, Field(description="Limit to these video ids. Omit to search every indexed video.")
]
Wait = Annotated[
    float, Field(ge=0, le=300, description="Seconds to wait for the job before returning its current state.")
]


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    view = {
        "job_id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "message": job["message"],
        "source": job["source"],
        "video_id": job["video_id"],
    }
    if job.get("error"):
        view["error"] = job["error"]
    if job.get("result") is not None:
        view["result"] = job["result"]
    if job["status"] not in TERMINAL:
        view["next"] = f"call video_job(job_id='{job['id']}', wait_seconds=60) to keep waiting"
    return view


def _video_view(video: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "uploader", "source", "source_kind", "webpage_url", "path", "duration", "width")
    keys += ("height", "fps", "has_audio", "language", "status", "segment_count", "passage_count", "frame_count")
    view = {key: video[key] for key in keys if key in video}
    view["length"] = retrieval.clock(video.get("duration") or 0)
    return view


def _jpeg(source: Path | bytes, max_px: int) -> bytes:
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


def render_transcript(segments: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "srt":
        blocks = [
            f"{i}\n{_srt_time(s['start'], ',')} --> {_srt_time(s['end'], ',')}\n{s['text']}\n"
            for i, s in enumerate(segments, start=1)
        ]
        return "\n".join(blocks)
    if fmt == "vtt":
        blocks = [f"{_srt_time(s['start'], '.')} --> {_srt_time(s['end'], '.')}\n{s['text']}\n" for s in segments]
        return "WEBVTT\n\n" + "\n".join(blocks)
    if fmt == "plain":
        return " ".join(s["text"] for s in segments)
    return "\n".join(f"[{retrieval.clock(s['start'])}] {s['text']}" for s in segments)


class App:
    """Everything the tools share. Built once per process."""

    def __init__(self, settings: Settings) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.store = Store(settings.db_path)
        self.store.interrupt_stale_jobs()
        self.pipeline = Pipeline(settings, self.store, lambda: make_embedder(settings))

    @property
    def embedder(self):  # noqa: ANN201
        return self.pipeline.embedder

    def video(self, video_id: str) -> dict[str, Any]:
        video = self.store.get_video(video_id.strip())
        if video is None:
            known = ", ".join(v["id"] for v in self.store.list_videos()[:8]) or "none indexed yet"
            raise VideoQAError("not_found", f"no video {video_id!r}; known ids: {known}")
        return video

    def scope(self, video_ids: list[str] | None) -> list[str] | None:
        if video_ids is None:
            return None
        return [self.video(video_id)["id"] for video_id in video_ids]

    def source_file(self, video: dict[str, Any]) -> Path:
        path = Path(video.get("path") or "")
        if not path.is_file():
            raise VideoQAError("source_missing", f"the source file for {video['id']} is gone: {path}")
        return path

    def output_dir(self, requested: str | None, default: Path) -> Path:
        if not requested:
            default.mkdir(parents=True, exist_ok=True)
            return default
        path = self.settings.check_path(requested, must_exist=False)
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def wait(self, job_id: str, seconds: float, ctx: Context | None) -> dict[str, Any]:
        async def tick(job: dict[str, Any]) -> None:
            if ctx is not None:
                await ctx.report_progress(float(job["progress"]), 1.0, job["message"])

        return _job_view(await self.pipeline.wait(job_id, seconds, tick))

    async def shutdown(self) -> None:
        await self.pipeline.shutdown()
        self.store.close()


def build_server(settings: Settings) -> tuple[MCPServer, App]:
    app = App(settings)
    mcp = MCPServer(
        name="distributed-video-qa",
        title="Distributed Video QA",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://github.com/arthurcolle/distributed-video-qa-mcp",
    )

    # ------------------------------------------------------------ status

    @mcp.tool(title="Server status", annotations=READ)
    async def video_status() -> dict[str, Any]:
        """Report what this server can do right now: tools found, transcription and embedding backends, index size."""
        tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
        try:
            import yt_dlp.version

            ytdlp = yt_dlp.version.__version__
        except Exception as exc:  # noqa: BLE001
            ytdlp = f"unavailable ({exc})"
        if settings.embed_backend == "jina" and not settings.embed_api_key:
            embedding = "unavailable (set JINA_API_KEY)"
        else:
            embedding = f"{settings.embed_model if settings.embed_backend == 'jina' else 'hash-test-embedder'}"
            embedding += f" at {settings.embed_dim} dimensions"
        return {
            "version": __version__,
            "ffmpeg": tools["ffmpeg"] or "missing",
            "ffprobe": tools["ffprobe"] or "missing",
            "yt_dlp": ytdlp,
            "js_runtimes": sorted(dl.js_runtimes()) or "none (YouTube will offer only low qualities)",
            "transcription": stt.describe_backend(settings),
            "embedding": embedding,
            "state_dir": str(settings.state_dir),
            "limits": {
                "max_concurrent_jobs": settings.max_concurrent_jobs,
                "cpu_workers": settings.cpu_workers,
                "max_duration_seconds": settings.max_duration,
                "max_frames_per_video": settings.max_frames,
            },
            "index": app.store.stats(),
        }

    # ------------------------------------------------------------ download

    @mcp.tool(title="List download options for a URL", annotations=READ_WEB)
    async def video_probe(
        url: Annotated[str, Field(description="Public http(s) link to a single video.")],
        all_formats: Annotated[bool, Field(description="Also return every raw format, not just the menu.")] = False,
    ) -> dict[str, Any]:
        """Show what a video link offers before downloading: title, length, and each quality with its size and
        whether that quality has sound on its own or needs the audio track merged in."""
        info = dl.summarize_formats(await dl.probe(url, settings))
        if not all_formats:
            info.pop("formats")
        return info

    @mcp.tool(title="Download a video or its audio", annotations=WRITE_WEB)
    async def video_download(
        url: Annotated[str, Field(description="Public http(s) link to a single video.")],
        quality: Annotated[str, Field(description="'best' or a maximum height such as '1080p' or '720p'.")] = "best",
        mode: Annotated[
            Literal["video+audio", "video_only", "audio_only"],
            Field(description="'video+audio' merges the best audio into the chosen video quality."),
        ] = "video+audio",
        container: Annotated[Literal["mp4", "mkv", "webm"], Field(description="File type for video.")] = "mp4",
        audio_format: Annotated[
            Literal["m4a", "mp3", "opus", "flac", "wav"], Field(description="File type when mode is 'audio_only'.")
        ] = "m4a",
        format_id: Annotated[
            str | None,
            Field(description="Exact format id from video_probe, e.g. '136' or '136+140'. Overrides quality."),
        ] = None,
        output_dir: Annotated[str | None, Field(description="Folder to save into. Defaults to the server's.")] = None,
        wait_seconds: Wait = 30,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Save a video link to a file. High qualities are normally served without sound; this fetches the video
        and audio streams in parallel fragments and merges them into one playable file. Returns a job."""
        dest = app.output_dir(output_dir, settings.downloads_dir)
        dl.parse_quality(quality)
        job = app.pipeline.start_download(
            url,
            quality=quality,
            mode=mode,
            container=container,
            audio_format=audio_format,
            format_id=format_id,
            output_dir=dest,
        )
        return await app.wait(job["id"], wait_seconds, ctx)

    # ------------------------------------------------------------ ingest and jobs

    @mcp.tool(title="Index a video for questions", annotations=WRITE_WEB)
    async def video_ingest(
        source: Annotated[str, Field(description="A video URL or an absolute path to a local media file.")],
        quality: Annotated[
            str | None, Field(description="Download height for URLs, e.g. '720p'. Higher reads small text better.")
        ] = None,
        language: Annotated[
            str | None, Field(description="Spoken language code such as 'en'. Omit to detect it.")
        ] = None,
        frame_interval: Annotated[
            float | None, Field(gt=0, description="Seconds between sampled frames. Scene cuts are added on top.")
        ] = None,
        max_frames: Annotated[int | None, Field(gt=0, description="Upper bound on frames kept for this video.")] = None,
        transcribe: Annotated[bool, Field(description="Transcribe the speech.")] = True,
        frames: Annotated[bool, Field(description="Sample and embed frames.")] = True,
        force: Annotated[bool, Field(description="Redo stages that already completed.")] = False,
        wait_seconds: Wait = 30,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Download (if a URL), transcribe, sample frames and embed everything so the video can be searched and
        questioned. Transcription and frame work run in parallel. Re-running on the same video reuses finished
        stages. Returns a job; poll video_job until it succeeds, then use the video_id in its result."""
        options = IngestOptions(
            quality=quality,
            language=language,
            frame_interval=frame_interval,
            max_frames=max_frames,
            transcribe=transcribe,
            frames=frames,
            force=force,
        )
        if quality:
            dl.parse_quality(quality)
        if not transcribe and not frames:
            raise VideoQAError("nothing_to_do", "enable transcribe, frames, or both")
        # Reject a bad source now instead of minutes into a job.
        if is_url(source):
            check_url(source, allow_private=settings.allow_private_hosts)
        elif not settings.check_path(source).is_file():
            raise VideoQAError("not_a_file", f"{source} is not a file")
        _ = app.embedder  # fail now, not minutes into the job, when no embedding backend is configured
        job = app.pipeline.start_ingest(source, options)
        return await app.wait(job["id"], wait_seconds, ctx)

    @mcp.tool(title="Check a job", annotations=READ)
    async def video_job(
        job_id: Annotated[str, Field(description="Job id returned by video_ingest or video_download.")],
        wait_seconds: Wait = 0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get a job's status, stage and progress, optionally waiting for it to finish."""
        return await app.wait(job_id, wait_seconds, ctx)

    @mcp.tool(title="List jobs", annotations=READ)
    async def video_jobs(
        active_only: Annotated[bool, Field(description="Only queued and running jobs.")] = False,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """List recent download and ingest jobs, newest first."""
        return {"jobs": [_job_view(job) for job in app.store.list_jobs(limit, active_only)]}

    @mcp.tool(title="Cancel a job", annotations=WRITE)
    async def video_cancel(
        job_id: Annotated[str, Field(description="Job id to stop.")],
    ) -> dict[str, Any]:
        """Stop a queued or running job. Work already finished stays in the index."""
        if app.store.get_job(job_id) is None:
            raise VideoQAError("not_found", f"no job {job_id}")
        signalled = app.pipeline.cancel(job_id)
        await asyncio.sleep(0.2)
        view = _job_view(app.store.get_job(job_id) or {})
        view["cancel_requested"] = signalled
        return view

    # ------------------------------------------------------------ library

    @mcp.tool(title="List indexed videos", annotations=READ)
    async def video_list() -> dict[str, Any]:
        """List every video in the index with its id, title, length and how much was indexed."""
        videos = [_video_view(video) for video in app.store.list_videos()]
        return {"count": len(videos), "videos": videos}

    @mcp.tool(title="Describe one video", annotations=READ)
    async def video_info(
        video_id: Annotated[str, Field(description="Id from video_list or an ingest result.")],
    ) -> dict[str, Any]:
        """Details for one video: source, dimensions, language, index counts and the uploader's description."""
        video = app.video(video_id)
        view = _video_view(video)
        view["counts"] = app.store.counts(video["id"])
        view["stages_done"] = sorted(video["stages"])
        view["upload_date"] = video["meta"].get("upload_date")
        view["description"] = video["meta"].get("description")
        return view

    @mcp.tool(title="Remove a video from the index", annotations=DESTROY)
    async def video_delete(
        video_id: Annotated[str, Field(description="Id of the video to remove.")],
        delete_files: Annotated[
            bool, Field(description="Also delete this server's downloaded copy and frames. Your own files stay.")
        ] = True,
    ) -> dict[str, Any]:
        """Delete a video's transcript, frames and vectors from the index. A local file you ingested by path is
        never deleted; only the server's own downloaded copy and extracted frames are."""
        video = app.video(video_id)
        removed = app.pipeline.delete(video["id"], delete_files=delete_files)
        return {"video_id": video["id"], "removed": removed, "files_deleted": delete_files}

    # ------------------------------------------------------------ reading

    @mcp.tool(title="Read the transcript", annotations=READ)
    async def video_transcript(
        video_id: Annotated[str, Field(description="Id of the video.")],
        start: Annotated[float | None, Field(ge=0, description="From this second.")] = None,
        end: Annotated[float | None, Field(ge=0, description="Up to this second.")] = None,
        format: Annotated[
            Literal["timestamped", "plain", "srt", "vtt", "json"], Field(description="Output layout.")
        ] = "timestamped",
        max_chars: Annotated[int, Field(ge=500, le=400_000, description="Truncate the text beyond this.")] = 60_000,
    ) -> dict[str, Any]:
        """Return what was said, for the whole video or a time range, as timestamped lines, plain text,
        SRT or WebVTT subtitles, or JSON segments."""
        video = app.video(video_id)
        segments = app.store.segments(video["id"], start, end)
        out: dict[str, Any] = {
            "video_id": video["id"],
            "title": video["title"],
            "language": video["language"],
            "segment_count": len(segments),
        }
        if not segments:
            out["note"] = "no transcript for this range; the video may have no speech or was indexed frames-only"
        if format == "json":
            kept: list[dict[str, Any]] = []
            used = 0
            for seg in segments:
                used += len(seg["text"]) + 40
                if used > max_chars:
                    break
                kept.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            out["segments"] = kept
            out["truncated"] = len(kept) < len(segments)
            return out
        text = render_transcript(segments, format)
        out["truncated"] = len(text) > max_chars
        out["text"] = text[:max_chars]
        if out["truncated"]:
            out["note"] = "truncated; pass start/end to page through the rest"
        return out

    # ------------------------------------------------------------ retrieval

    async def _images(paths: list[tuple[str, Path | bytes]], max_px: int) -> list[Any]:
        blocks: list[Any] = []
        for label, source in paths:
            try:
                data = await asyncio.to_thread(_jpeg, source, max_px)
            except OSError:
                blocks.append(f"{label}: image file is missing")
                continue
            blocks.append(label)
            blocks.append(Image(data=data, format="jpeg"))
        return blocks

    @mcp.tool(title="Search inside videos", annotations=READ)
    async def video_search(
        query: Annotated[str, Field(min_length=1, description="What to look for, in natural language.")],
        video_ids: VideoIds = None,
        modalities: Annotated[
            list[Literal["transcript", "frames"]],
            Field(description="Search what was said, what was shown, or both."),
        ] = ["transcript", "frames"],  # noqa: B006
        top_k: Annotated[int, Field(ge=1, le=50)] = 10,
        start: Annotated[float | None, Field(ge=0, description="Only after this second.")] = None,
        end: Annotated[float | None, Field(ge=0, description="Only before this second.")] = None,
    ) -> dict[str, Any]:
        """Ranked moments matching a query. Combines meaning-based search over the transcript, keyword search,
        and text-to-image search over frames, so 'the slide with the pricing table' finds the frame even if
        nobody said those words. Each hit says which of those matched it."""
        hits = await retrieval.search(
            app.store,
            app.embedder,
            query,
            video_ids=app.scope(video_ids),
            modalities=modalities,
            top_k=top_k,
            start=start,
            end=end,
        )
        return {"query": query, "count": len(hits), "hits": [hit.to_dict() for hit in hits]}

    @mcp.tool(title="Find frames that look like an image", annotations=READ)
    async def video_search_by_image(
        image_path: Annotated[str, Field(description="Absolute path to a picture (jpg, png, webp).")],
        video_ids: VideoIds = None,
        top_k: Annotated[int, Field(ge=1, le=50)] = 8,
    ) -> dict[str, Any]:
        """Find where a picture, logo, chart or person-free scene appears: frames that look like the image, and
        transcript passages about what it shows."""
        path = settings.check_path(image_path)
        hits = await retrieval.search_by_image(
            app.store, app.embedder, path, video_ids=app.scope(video_ids), top_k=top_k
        )
        return {"image": str(path), "count": len(hits), "hits": [hit.to_dict() for hit in hits]}

    @mcp.tool(title="Gather evidence for a question", annotations=READ, structured_output=False)
    async def video_ask(
        question: Annotated[str, Field(min_length=1, description="The question to answer from the video(s).")],
        video_ids: VideoIds = None,
        max_moments: Annotated[int, Field(ge=1, le=12, description="How many moments to return.")] = 5,
        frames_per_moment: Annotated[int, Field(ge=0, le=4, description="Images attached per moment.")] = 2,
        image_px: Annotated[int, Field(ge=256, le=1568, description="Long side of returned images.")] = 768,
    ) -> list[Any]:
        """Retrieve the moments most relevant to a question and return them as evidence: for each moment the
        transcript around it with timestamps, and the frames from that moment as images. Read the evidence,
        then answer and cite timestamps. If the evidence does not cover the question, say so or search again
        with different wording."""
        scope = app.scope(video_ids)
        hits = await retrieval.search(
            app.store, app.embedder, question, video_ids=scope, top_k=max(12, max_moments * 4)
        )
        moments = retrieval.build_moments(app.store, hits, max_moments=max_moments, frames_per_moment=frames_per_moment)
        titles = {m.video_id: (app.store.get_video(m.video_id) or {}).get("title") for m in moments}
        payload = {
            "question": question,
            "moments": [{**m.to_dict(), "title": titles[m.video_id]} for m in moments],
            "guidance": "Answer only from this evidence and cite times as m:ss. Images follow in moment order.",
        }
        if not moments:
            payload["guidance"] = "Nothing relevant was retrieved. Check video_list, or rephrase the question."
        images = [
            (f"Frame from {moment.video_id} at {retrieval.clock(frame['t'])}", Path(frame["path"]))
            for moment in moments
            for frame in moment.frames
        ]
        return [json.dumps(payload, indent=2, ensure_ascii=False), *await _images(images, image_px)]

    @mcp.tool(title="Look at frames", annotations=READ, structured_output=False)
    async def video_frames(
        video_id: Annotated[str, Field(description="Id of the video.")],
        timestamps: Annotated[
            list[float] | None, Field(description="Exact seconds to look at. Up to 12.", max_length=12)
        ] = None,
        start: Annotated[float | None, Field(ge=0, description="Or: evenly spaced frames from this second...")] = None,
        end: Annotated[float | None, Field(ge=0, description="...to this second.")] = None,
        count: Annotated[int, Field(ge=1, le=12, description="How many frames across start..end.")] = 6,
        exact: Annotated[
            bool,
            Field(description="Decode the frame at the precise time from the source instead of the nearest sample."),
        ] = True,
        image_px: Annotated[int, Field(ge=256, le=1568, description="Long side of returned images.")] = 1024,
    ) -> list[Any]:
        """Return frames as images, at specific timestamps or spread across a range. With exact=true each frame
        is decoded from the source file at that instant, at higher resolution than the indexed samples, which
        is what you want for reading small on-screen text."""
        video = app.video(video_id)
        duration = float(video.get("duration") or 0.0)
        if timestamps:
            times = [min(max(0.0, t), max(duration - 0.05, 0.0)) for t in timestamps]
        else:
            lo = start or 0.0
            hi = min(end if end is not None else duration, duration)
            if hi <= lo:
                raise VideoQAError("bad_range", f"end must be after start and within the video ({duration:.1f}s)")
            times = [lo + (hi - lo) * (i + 0.5) / count for i in range(count)]
        sources: list[tuple[str, Path | bytes]] = []
        if exact:
            source = app.source_file(video)
            gate = asyncio.Semaphore(settings.cpu_workers)

            async def grab(t: float) -> bytes:
                async with gate:
                    return await media.frame_at(source, t, width=max(image_px, 1280))

            decoded = await asyncio.gather(*(grab(t) for t in times))
            sources = [
                (f"Frame at {retrieval.clock(t)} ({t:.2f}s)", data) for t, data in zip(times, decoded, strict=True)
            ]
        else:
            seen: set[str] = set()
            for t in times:
                frame = app.store.nearest_frame(video["id"], t)
                if frame and frame["path"] not in seen:
                    seen.add(frame["path"])
                    sources.append(
                        (f"Sampled frame at {retrieval.clock(frame['t'])} ({frame['t']:.2f}s)", Path(frame["path"]))
                    )
        header = {"video_id": video["id"], "title": video["title"], "frames": len(sources), "exact": exact}
        return [json.dumps(header), *await _images(sources, image_px)]

    @mcp.tool(title="Outline a video", annotations=READ, structured_output=False)
    async def video_chapters(
        video_id: Annotated[str, Field(description="Id of the video.")],
        include_images: Annotated[bool, Field(description="Attach one frame per chapter.")] = False,
        image_px: Annotated[int, Field(ge=256, le=1024)] = 512,
    ) -> list[Any]:
        """Split a video into chapters. Uses the uploader's chapters when the source has them; otherwise finds
        topic shifts from the transcript embeddings (or from frames for silent video). Each chapter has its
        time span, opening words and distinguishing keywords, which is enough to title and summarise it."""
        video = app.video(video_id)
        found = retrieval.chapters(app.store, app.embedder, video)
        payload = {"video_id": video["id"], "title": video["title"], "chapters": found}
        blocks: list[Any] = [json.dumps(payload, indent=2, ensure_ascii=False)]
        if include_images:
            images = [
                (f"Chapter {c['index']} at {retrieval.clock(c['start'])}", Path(c["frame_path"]))
                for c in found[:12]
                if c.get("frame_path")
            ]
            blocks += await _images(images, image_px)
        return blocks

    @mcp.tool(title="Export a clip", annotations=WRITE)
    async def video_clip(
        video_id: Annotated[str, Field(description="Id of the video.")],
        start: Annotated[float, Field(ge=0, description="Clip start in seconds.")],
        end: Annotated[float, Field(gt=0, description="Clip end in seconds.")],
        exact: Annotated[
            bool, Field(description="Re-encode for frame-accurate cuts. Off is instant but snaps to keyframes.")
        ] = False,
        output_dir: Annotated[str | None, Field(description="Folder to save into. Defaults to the server's.")] = None,
    ) -> dict[str, Any]:
        """Cut a time range out of a video into its own MP4 file, with audio."""
        video = app.video(video_id)
        if end <= start:
            raise VideoQAError("bad_range", "end must be after start")
        if end - start > 3600:
            raise VideoQAError("bad_range", "clips are limited to one hour")
        source = app.source_file(video)
        dest_dir = app.output_dir(output_dir, settings.downloads_dir / "clips")
        dest = dest_dir / f"{video['id']}_{int(start * 1000)}_{int(end * 1000)}.mp4"
        await media.cut_clip(source, start, end, dest, reencode=exact)
        info = await media.probe(dest)
        return {
            "path": str(dest),
            "duration": round(info.duration, 2),
            "size_bytes": info.size_bytes,
            "has_audio": info.has_audio,
            "exact": exact,
        }

    return mcp, app
