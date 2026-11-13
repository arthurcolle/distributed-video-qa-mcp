"""Background jobs: download, and the ingest pipeline that turns a video into an index.

Ingest runs its two expensive branches at the same time:

    source -> probe -+-> audio -> transcript -> passages -> text vectors  --+
                     |                                                      +-> ready
                     +-> frames (parallel segments) -> dedupe -> image vectors

Each branch records a signature of its inputs when it finishes, so a retry or a
restart skips whatever already completed with the same settings.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import download as dl
from . import media
from . import transcribe as stt
from .config import Settings
from .embed import PASSAGE, Embedder
from .errors import Cancelled, VideoQAError
from .netguard import is_url
from .store import Store

log = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_AUTO = object()


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


@dataclass
class IngestOptions:
    quality: str | None = None
    language: str | None = None
    frame_interval: float | None = None
    max_frames: int | None = None
    scene_threshold: float | None = None
    transcribe: bool = True
    frames: bool = True
    force: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _Progress:
    """Maps per-stage progress onto one 0..1 number for the job row."""

    store: Store
    job_id: str
    weights: dict[str, float]
    done: dict[str, float] = field(default_factory=dict)
    _last_write: float = 0.0

    def update(self, stage: str, fraction: float, message: str, *, force: bool = False) -> None:
        self.done[stage] = max(0.0, min(1.0, fraction))
        now = time.monotonic()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        total = sum(self.weights.values()) or 1.0
        overall = sum(self.weights[name] * self.done.get(name, 0.0) for name in self.weights) / total
        self.store.update_job(self.job_id, stage=stage, progress=round(overall, 4), message=message)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        embedder: Embedder | Callable[[], Embedder],
        transcriber: stt.Transcriber | None | object = _AUTO,
    ) -> None:
        self.settings = settings
        self.store = store
        self._embedder = embedder
        self._transcriber = transcriber
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._slots = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._video_locks: dict[str, asyncio.Lock] = {}

    @property
    def embedder(self) -> Embedder:
        """Built on first use so that downloads work without an embedding key."""
        if callable(self._embedder):
            self._embedder = self._embedder()
        return self._embedder

    def transcriber(self) -> stt.Transcriber | None:
        if self._transcriber is _AUTO:
            self._transcriber = stt.resolve_backend(self.settings)
        return self._transcriber  # type: ignore[return-value]

    # ------------------------------------------------------------ job control

    def _start(
        self, kind: str, source: str, params: dict[str, Any], work: Callable[[str, threading.Event], Awaitable[dict]]
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        self.store.create_job(job_id, kind, source, params)
        cancel = threading.Event()
        self._cancel[job_id] = cancel

        async def runner() -> None:
            try:
                async with self._slots:
                    self.store.update_job(job_id, status="running", message="starting")
                    result = await work(job_id, cancel)
                self.store.update_job(
                    job_id, status="succeeded", progress=1.0, stage="done", message="done", result=result
                )
            except (asyncio.CancelledError, Cancelled):
                self.store.update_job(job_id, status="cancelled", message="cancelled")
            except VideoQAError as exc:
                self.store.update_job(job_id, status="failed", error=str(exc), message=exc.message)
            except Exception as exc:  # noqa: BLE001 - a job must never take the server down
                log.exception("job %s crashed", job_id)
                self.store.update_job(
                    job_id, status="failed", error=f"internal: {type(exc).__name__}: {exc}", message="internal error"
                )
            finally:
                self._tasks.pop(job_id, None)
                self._cancel.pop(job_id, None)

        self._tasks[job_id] = asyncio.create_task(runner(), name=f"job-{job_id}")
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    async def wait(
        self, job_id: str, timeout: float, on_tick: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    ) -> dict[str, Any]:
        """Return the job once it finishes or after `timeout` seconds, whichever is first."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            job = self.store.get_job(job_id)
            if job is None:
                raise VideoQAError("not_found", f"no job {job_id}")
            if job["status"] in TERMINAL or time.monotonic() >= deadline:
                return job
            if on_tick:
                with contextlib.suppress(Exception):
                    await on_tick(job)
            await asyncio.sleep(0.5)

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        task = self._tasks.get(job_id)
        if event:
            event.set()
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def shutdown(self) -> None:
        for job_id in list(self._tasks):
            self.cancel(job_id)
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if not callable(self._embedder):
            await self._embedder.aclose()

    # ------------------------------------------------------------ download

    def start_download(
        self,
        url: str,
        *,
        quality: str,
        mode: str,
        container: str,
        audio_format: str,
        format_id: str | None,
        output_dir: Path | None,
    ) -> dict[str, Any]:
        dest = output_dir or self.settings.downloads_dir
        params = {"quality": quality, "mode": mode, "container": container, "format_id": format_id}

        async def work(job_id: str, cancel: threading.Event) -> dict[str, Any]:
            progress = _Progress(self.store, job_id, {"download": 1.0})
            result = await dl.download(
                url,
                dest,
                self.settings,
                quality=quality,
                mode=mode,
                container=container,
                audio_format=audio_format,
                format_id=format_id,
                on_progress=lambda f, m: progress.update("download", f, m),
                cancel=cancel,
            )
            info = await media.probe(result.path)
            return {
                "path": str(result.path),
                "title": result.info.get("title"),
                "mode": mode,
                "audio_merged": result.merged,
                "has_audio": info.has_audio,
                "has_video": info.has_video,
                "width": info.width,
                "height": info.height,
                "duration": info.duration,
                "size_bytes": info.size_bytes,
                "video_codec": info.video_codec,
                "audio_codec": info.audio_codec,
            }

        return self._start("download", url, params, work)

    # ------------------------------------------------------------ ingest

    def start_ingest(self, source: str, options: IngestOptions) -> dict[str, Any]:
        async def work(job_id: str, cancel: threading.Event) -> dict[str, Any]:
            return await self._ingest(job_id, cancel, source, options)

        return self._start("ingest", source, options.to_dict(), work)

    async def _resolve(
        self, job_id: str, cancel: threading.Event, source: str, options: IngestOptions, progress: _Progress
    ) -> tuple[str, Path, dict[str, Any]]:
        """Find or fetch the media file and decide the video id."""
        if is_url(source):
            progress.update("download", 0.0, "reading video metadata", force=True)
            info = await dl.probe(source, self.settings)
            video_id = _digest("url", info.get("extractor_key"), info.get("id"))[:12]
            self.store.update_job(job_id, video_id=video_id)
            existing = self.store.get_video(video_id)
            if existing and existing.get("path") and Path(existing["path"]).exists() and not options.force:
                path = Path(existing["path"])
            else:
                quality = options.quality or f"{self.settings.ingest_height}p"
                result = await dl.download(
                    source,
                    self.settings.video_dir(video_id) / "source",
                    self.settings,
                    quality=quality,
                    on_progress=lambda f, m: progress.update("download", f, m),
                    cancel=cancel,
                )
                path = result.path
            fields = {
                "source": source,
                "source_kind": "url",
                "title": info.get("title"),
                "uploader": info.get("uploader") or info.get("channel"),
                "webpage_url": info.get("webpage_url"),
                "meta": {
                    "extractor": info.get("extractor_key"),
                    "remote_id": info.get("id"),
                    "upload_date": info.get("upload_date"),
                    "description": (info.get("description") or "")[:2000],
                    "chapters": info.get("chapters") or [],
                },
            }
        else:
            path = self.settings.check_path(source)
            if not path.is_file():
                raise VideoQAError("not_a_file", f"{path} is not a file")
            progress.update("download", 0.2, "hashing local file", force=True)
            video_id = _digest("file", await asyncio.to_thread(_file_digest, path))[:12]
            self.store.update_job(job_id, video_id=video_id)
            fields = {"source": str(path), "source_kind": "file", "title": path.stem, "meta": {}}
        progress.update("download", 1.0, "source ready", force=True)
        return video_id, path, fields

    async def _ingest(
        self, job_id: str, cancel: threading.Event, source: str, options: IngestOptions
    ) -> dict[str, Any]:
        started = time.monotonic()
        settings = self.settings
        weights = {"download": 0.15, "transcribe": 0.45, "frames": 0.2, "embed": 0.2}
        progress = _Progress(self.store, job_id, weights)
        video_id, path, fields = await self._resolve(job_id, cancel, source, options, progress)

        lock = self._video_locks.setdefault(video_id, asyncio.Lock())
        async with lock:
            info = await media.probe(path)
            if info.duration > settings.max_duration:
                raise VideoQAError(
                    "too_long", f"video is {info.duration:.0f}s; the limit is DVQA_MAX_DURATION={settings.max_duration}"
                )
            video = self.store.upsert_video(
                video_id,
                path=str(path),
                duration=info.duration,
                width=info.width,
                height=info.height,
                fps=info.fps,
                has_audio=int(info.has_audio),
                status="processing",
                **fields,
            )
            if options.force:
                self.store.set_stage(video_id, "transcript", None)
                self.store.set_stage(video_id, "frames", None)
                video = self.store.get_video(video_id) or video
            stages: dict[str, str] = video["stages"]
            workdir = settings.video_dir(video_id)
            workdir.mkdir(parents=True, exist_ok=True)
            tokens_before = self.embedder.tokens_used
            report: dict[str, Any] = {}
            embed_parts: dict[str, float] = {}

            def embed_progress(part: str, share: float) -> Callable[[int, int], None]:
                def update(done: int, total: int) -> None:
                    embed_parts[part] = share * done / max(total, 1)
                    progress.update("embed", sum(embed_parts.values()), f"embedding {part} {done}/{total}")

                return update

            do_frames = options.frames and info.has_video
            do_speech = options.transcribe and info.has_audio and settings.transcribe_backend != "none"
            text_share = 0.3 if do_frames else 1.0
            frame_share = (1.0 - text_share) if do_speech else 1.0

            async def speech_branch() -> None:
                if not do_speech:
                    progress.update("transcribe", 1.0, "no speech track to transcribe")
                    embed_parts["transcript"] = text_share
                    report["transcript"] = "skipped"
                    return
                backend = self.transcriber()
                assert backend is not None
                signature = _digest(
                    "transcript", backend.name, options.language, self.embedder.model, self.embedder.dim
                )
                if stages.get("transcript") == signature:
                    progress.update("transcribe", 1.0, "transcript already indexed")
                    embed_parts["transcript"] = text_share
                    report["transcript"] = "cached"
                    return
                audio = workdir / "audio.flac"
                progress.update("transcribe", 0.02, "extracting audio", force=True)
                await media.extract_audio(path, audio)
                try:
                    transcript = await stt.transcribe(
                        audio,
                        info.duration,
                        workdir,
                        settings,
                        language=options.language,
                        backend=backend,
                        on_progress=lambda f, m: progress.update("transcribe", 0.05 + 0.95 * f, m),
                    )
                finally:
                    audio.unlink(missing_ok=True)
                progress.update("transcribe", 1.0, f"{len(transcript.segments)} segments transcribed", force=True)
                passages = stt.build_passages(transcript.segments)
                vectors = await self.embedder.embed_texts(
                    [p.text for p in passages], PASSAGE, embed_progress("transcript", text_share)
                )
                await asyncio.to_thread(
                    self.store.replace_transcript,
                    video_id,
                    [(s.start, s.end, s.text) for s in transcript.segments],
                    [(p.start, p.end, p.text) for p in passages],
                    vectors,
                    self.embedder.model,
                )
                self.store.upsert_video(video_id, language=transcript.language)
                self.store.set_stage(video_id, "transcript", signature)
                report["transcript"] = f"{len(transcript.segments)} segments via {transcript.backend}"

            async def frame_branch() -> None:
                if not do_frames:
                    progress.update("frames", 1.0, "no frames requested")
                    embed_parts["frames"] = frame_share
                    report["frames"] = "skipped"
                    return
                interval = options.frame_interval or settings.frame_interval
                limit = options.max_frames or settings.max_frames
                threshold = options.scene_threshold or settings.scene_threshold
                signature = _digest(
                    "frames", interval, limit, threshold, settings.frame_width, settings.dedupe_distance,
                    settings.dedupe_max_gap, self.embedder.model, self.embedder.dim, settings.embed_image_px,
                )  # fmt: skip
                if stages.get("frames") == signature:
                    progress.update("frames", 1.0, "frames already indexed")
                    embed_parts["frames"] = frame_share
                    report["frames"] = "cached"
                    return
                progress.update("frames", 0.02, "sampling frames", force=True)
                staging = workdir / "frames.tmp"
                frames = await media.extract_frames(
                    path,
                    staging,
                    info.duration,
                    interval=interval,
                    scene_threshold=threshold,
                    scene_min_gap=settings.scene_min_gap,
                    width=settings.frame_width,
                    workers=settings.cpu_workers,
                    max_frames=limit,
                )
                sampled = len(frames)
                progress.update("frames", 0.7, f"{sampled} frames sampled, removing near-duplicates", force=True)
                await media.hash_frames(frames, settings.cpu_workers * 2)
                frames, dupes = media.dedupe(frames, distance=settings.dedupe_distance, max_gap=settings.dedupe_max_gap)
                frames, extra = media.thin(frames, limit)
                for frame in (*dupes, *extra):
                    frame.path.unlink(missing_ok=True)
                final = workdir / "frames"
                if final.exists():
                    shutil.rmtree(final)
                final.mkdir(parents=True)
                for index, frame in enumerate(frames):
                    target = final / f"{index:05d}_{int(frame.t * 1000):09d}.jpg"
                    frame.path.rename(target)
                    frame.path = target
                shutil.rmtree(staging, ignore_errors=True)
                progress.update("frames", 1.0, f"{len(frames)} distinct frames kept", force=True)
                vectors = await self.embedder.embed_images(
                    [f.path for f in frames], PASSAGE, embed_progress("frames", frame_share)
                )
                await asyncio.to_thread(
                    self.store.replace_frames,
                    video_id,
                    [(f.t, str(f.path), f.scene_cut, f.phash) for f in frames],
                    vectors,
                    self.embedder.model,
                )
                self.store.set_stage(video_id, "frames", signature)
                report["frames"] = f"{len(frames)} kept of {sampled} sampled"

            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(speech_branch())
                    group.create_task(frame_branch())
            except BaseException as exc:
                self.store.upsert_video(video_id, status="failed")
                if isinstance(exc, BaseExceptionGroup):
                    raise exc.exceptions[0] from None
                raise

            counts = self.store.counts(video_id)
            if counts.get("vectors", 0) == 0:
                self.store.upsert_video(video_id, status="failed")
                raise VideoQAError("nothing_indexed", "the video produced no transcript and no frames to index")
            self.store.upsert_video(video_id, status="ready")
            progress.update("embed", 1.0, "indexed", force=True)
            final_video = self.store.get_video(video_id) or {}
            return {
                "video_id": video_id,
                "title": final_video.get("title"),
                "duration": info.duration,
                "language": final_video.get("language"),
                "counts": counts,
                "stages": report,
                "embedding_model": self.embedder.model,
                "embedding_tokens": self.embedder.tokens_used - tokens_before,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }

    # ------------------------------------------------------------ removal

    def delete(self, video_id: str, *, delete_files: bool = True) -> bool:
        removed = self.store.delete_video(video_id)
        if delete_files:
            shutil.rmtree(self.settings.video_dir(video_id), ignore_errors=True)
        return removed
