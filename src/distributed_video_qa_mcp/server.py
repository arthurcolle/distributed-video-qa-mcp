"""The MCP tool surface.

Nineteen tools in six groups: acquire, index, jobs, retrieve, inspect and
library. Every tool declares a typed result (see `models.py`), published as its
output schema and validated on the way out. The server retrieves evidence; the
model calling it does the reasoning. A question comes back as timestamped
transcript plus the actual frames as images, so the caller reads the screen
itself instead of trusting a caption.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from . import __version__, media, retrieval
from . import download as dl
from .app import App, job_model, render_transcript, video_summary
from .config import Settings
from .errors import VideoQAError
from .models import (
    AskResult,
    BatchResult,
    CancelResult,
    ChaptersResult,
    ClipResult,
    DeleteResult,
    FramesResult,
    Job,
    JobList,
    PhraseResult,
    ProbeResult,
    SearchResult,
    StatusReport,
    TranscriptResult,
    VideoDetail,
    VideoList,
)
from .netguard import check_url, is_url
from .pipeline import IngestOptions
from .prompts import register_prompts
from .resources import register_resources
from .verify import SUPPORT_THRESHOLD

INSTRUCTIONS = """\
Download, index and question video. Every tool returns typed structured content.
Error text reads 'Error executing tool <tool>: <code>: <message> — <hint>'; branch on <code>.

Flow:
1. video_ingest(source) with a URL or local path (video_ingest_batch for several, or a playlist). It returns a \
Job; call video_job(job_id, wait_seconds=60) until status is "succeeded". The result carries the video_id.
2. video_ask(question) returns the most relevant moments: timestamped transcript plus the frames as images. \
Answer from that evidence and cite timestamps as m:ss. If evidence.best_* similarities are low and \
keyword_matches is 0, say the video does not cover it.
3. Narrower tools: video_search (ranked hits), video_find_phrase (every exact occurrence), video_transcript \
(exact wording), video_frames (look at any instant, full resolution), video_chapters (structure), video_clip \
(export a segment).

To only save a file: video_probe(url) shows each quality and whether it needs an audio merge; video_download(url) \
merges the best audio in by default.
"""

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
READ_WEB = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITE_WEB = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTROY = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)

VideoId = Annotated[str, Field(description="Video id from video_list or an ingest result.")]
VideoIds = Annotated[list[str] | None, Field(description="Limit to these video ids. Omit for every indexed video.")]
Wait = Annotated[float, Field(ge=0, le=300, description="Seconds to wait for completion before returning the job.")]
ImagePx = Annotated[int, Field(ge=256, le=1568, description="Long side of returned images, in pixels.")]
Quality = Annotated[str, Field(description="'best' or a maximum height such as '1080p' or '720p'.")]
OutputDir = Annotated[str | None, Field(description="Folder to write into (under DVQA_ALLOWED_ROOTS).")]


def _with_images(model: object, blocks: list) -> CallToolResult:
    """Structured result plus its JSON as text, followed by caption/image pairs."""
    data = model.model_dump(mode="json")  # type: ignore[attr-defined]
    text = TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False))
    return CallToolResult(content=[text, *blocks], structured_content=data)


def _ingest_options(
    quality: str | None,
    language: str | None,
    frame_interval: float | None,
    max_frames: int | None,
    transcribe: bool,
    frames: bool,
    force: bool,
) -> IngestOptions:
    if quality:
        dl.parse_quality(quality)
    if not transcribe and not frames:
        raise VideoQAError("nothing_to_do", "both transcribe and frames are off")
    return IngestOptions(
        quality=quality,
        language=language,
        frame_interval=frame_interval,
        max_frames=max_frames,
        transcribe=transcribe,
        frames=frames,
        force=force,
    )


def build_server(settings: Settings) -> tuple[MCPServer, App]:
    app = App(settings)
    mcp = MCPServer(
        name="distributed-video-qa",
        title="Distributed Video QA",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://github.com/arthurcolle/distributed-video-qa-mcp",
        log_level="WARNING",
    )

    def check_source(source: str) -> None:
        """Reject a bad source now instead of minutes into a job."""
        if is_url(source):
            check_url(source, allow_private=settings.allow_private_hosts)
        elif not settings.check_path(source).is_file():
            raise VideoQAError("not_a_file", f"{source} is not a file")

    # ================================================================ status

    @mcp.tool(title="Server status", annotations=READ)
    async def video_status() -> StatusReport:
        """Readiness and capacity: FFmpeg, yt-dlp and its JavaScript runtimes, the transcription and embedding
        backends, limits, and index size. `ready` is false, with reasons in `problems`, when ingest would fail."""
        return app.status()

    # ================================================================ acquire

    @mcp.tool(title="List download options for a URL", annotations=READ_WEB)
    async def video_probe(
        url: Annotated[str, Field(description="Public http(s) link to a single video.")],
        all_formats: Annotated[bool, Field(description="Also return every raw stream.")] = False,
    ) -> ProbeResult:
        """Show what a link offers before downloading: title, length, and each quality with its size and whether
        it has sound on its own or needs the audio track merged in (most qualities above 360p do)."""
        info = dl.summarize_formats(await dl.probe(url, settings))
        if not all_formats:
            info["formats"] = None
        return ProbeResult.model_validate(info)

    @mcp.tool(title="Download a video or its audio", annotations=WRITE_WEB)
    async def video_download(
        url: Annotated[str, Field(description="Public http(s) link to a single video.")],
        quality: Quality = "best",
        mode: Annotated[
            Literal["video+audio", "video_only", "audio_only"],
            Field(description="'video+audio' merges the best audio track into the chosen video quality."),
        ] = "video+audio",
        container: Annotated[Literal["mp4", "mkv", "webm"], Field(description="Video file type.")] = "mp4",
        audio_format: Annotated[
            Literal["m4a", "mp3", "opus", "flac", "wav"], Field(description="File type for audio_only.")
        ] = "m4a",
        format_id: Annotated[
            str | None, Field(description="Exact id from video_probe, e.g. '136' or '136+140'. Overrides quality.")
        ] = None,
        output_dir: OutputDir = None,
        wait_seconds: Wait = 30,
        ctx: Context | None = None,
    ) -> Job:
        """Save a link to a file. Fetches video and audio streams in parallel fragments and muxes them into one
        playable file, preferring H.264/AAC for MP4 so it opens everywhere. Returns a Job whose result is a
        DownloadReport."""
        dl.parse_quality(quality)
        check_url(url, allow_private=settings.allow_private_hosts)
        job = app.pipeline.start_download(
            url,
            quality=quality,
            mode=mode,
            container=container,
            audio_format=audio_format,
            format_id=format_id,
            output_dir=app.output_dir(output_dir, settings.downloads_dir),
        )
        return await app.wait(job["id"], wait_seconds, ctx)

    # ================================================================ index

    @mcp.tool(title="Index a video for questions", annotations=WRITE_WEB)
    async def video_ingest(
        source: Annotated[str, Field(description="A video URL or the absolute path of a local media file.")],
        quality: Annotated[
            str | None, Field(description="Download height for URLs; higher reads small text better.")
        ] = None,
        language: Annotated[str | None, Field(description="Spoken language code such as 'en'. Omit to detect.")] = None,
        frame_interval: Annotated[
            float | None, Field(gt=0, description="Seconds between frames; cuts are added.")
        ] = None,
        max_frames: Annotated[int | None, Field(gt=0, description="Frames kept after de-duplication.")] = None,
        transcribe: Annotated[bool, Field(description="Transcribe the speech.")] = True,
        frames: Annotated[bool, Field(description="Sample and embed frames.")] = True,
        force: Annotated[bool, Field(description="Redo stages that already completed.")] = False,
        wait_seconds: Wait = 30,
        ctx: Context | None = None,
    ) -> Job:
        """Download (for a URL), transcribe, sample frames and embed, so the video can be searched and questioned.
        Speech and frame work run in parallel. Re-ingesting reuses finished stages, so a failed or interrupted
        job resumes. Returns a Job; its result holds the video_id, counts and timings."""
        options = _ingest_options(quality, language, frame_interval, max_frames, transcribe, frames, force)
        check_source(source)
        _ = app.embedder  # fail now, not minutes in, when no embedding backend is configured
        job = app.pipeline.start_ingest(source, options)
        return await app.wait(job["id"], wait_seconds, ctx)

    @mcp.tool(title="Index several videos or a playlist", annotations=WRITE_WEB)
    async def video_ingest_batch(
        sources: Annotated[
            list[str], Field(min_length=1, max_length=50, description="URLs or local paths; playlists expand.")
        ],
        playlist_limit: Annotated[int, Field(ge=1, le=200, description="Videos taken from each playlist.")] = 25,
        quality: Annotated[str | None, Field(description="Download height for URLs.")] = None,
        language: Annotated[str | None, Field(description="Spoken language code. Omit to detect.")] = None,
        transcribe: Annotated[bool, Field(description="Transcribe the speech.")] = True,
        frames: Annotated[bool, Field(description="Sample and embed frames.")] = True,
        wait_seconds: Wait = 0,
        ctx: Context | None = None,
    ) -> BatchResult:
        """Start one ingest job per source. Playlist and channel URLs expand to their first `playlist_limit`
        videos. Jobs run DVQA_MAX_CONCURRENT_JOBS at a time and the rest queue; a bad source is reported in
        `skipped` without stopping the others."""
        options = _ingest_options(quality, language, None, None, transcribe, frames, False)
        _ = app.embedder
        targets: list[str] = []
        expanded: list[str] = []
        skipped: list[str] = []
        for source in dict.fromkeys(s.strip() for s in sources if s.strip()):
            try:
                if is_url(source):
                    urls, was_playlist = await dl.expand(source, settings, limit=playlist_limit)
                    if was_playlist:
                        expanded.append(f"{source} -> {len(urls)} videos")
                    targets.extend(urls)
                else:
                    check_source(source)
                    targets.append(source)
            except VideoQAError as exc:
                skipped.append(f"{source}: {exc.code}: {exc.message}")
        targets = list(dict.fromkeys(targets))
        if len(targets) > 200:
            raise VideoQAError("too_many", f"{len(targets)} videos after expansion; the limit is 200 per batch")
        job_ids = [app.pipeline.start_ingest(target, options)["id"] for target in targets]
        jobs = await app.wait_all(job_ids, wait_seconds, ctx) if job_ids else []
        return BatchResult(jobs=jobs, expanded=expanded, skipped=skipped)

    # ================================================================ jobs

    @mcp.tool(title="Check a job", annotations=READ)
    async def video_job(
        job_id: Annotated[str, Field(description="Job id from video_ingest, video_ingest_batch or video_download.")],
        wait_seconds: Wait = 0,
        ctx: Context | None = None,
    ) -> Job:
        """A job's status, stage and progress, optionally waiting for it to finish. Progress is also streamed
        as MCP progress notifications while waiting."""
        return await app.wait(job_id, wait_seconds, ctx)

    @mcp.tool(title="List jobs", annotations=READ)
    async def video_jobs(
        active_only: Annotated[bool, Field(description="Only queued and running jobs.")] = False,
        limit: Annotated[int, Field(ge=1, le=100, description="Most jobs to return.")] = 20,
    ) -> JobList:
        """Recent download and ingest jobs, newest first."""
        return JobList(jobs=[job_model(job) for job in app.store.list_jobs(limit, active_only)])

    @mcp.tool(title="Cancel a job", annotations=WRITE)
    async def video_cancel(job_id: Annotated[str, Field(description="Job id to stop.")]) -> CancelResult:
        """Stop a queued or running job. Stages that already finished stay in the index and are reused."""
        if app.store.get_job(job_id) is None:
            raise VideoQAError("not_found", f"no job {job_id}")
        signalled = app.pipeline.cancel(job_id)
        await asyncio.sleep(0.2)
        job = job_model(app.store.get_job(job_id) or {})
        return CancelResult(**job.model_dump(), cancel_requested=signalled)

    # ================================================================ retrieve

    @mcp.tool(title="Gather evidence for a question", annotations=READ)
    async def video_ask(
        question: Annotated[str, Field(min_length=1, description="The question to answer from the video(s).")],
        video_ids: VideoIds = None,
        max_moments: Annotated[int, Field(ge=1, le=12, description="Moments to return.")] = 5,
        frames_per_moment: Annotated[int, Field(ge=0, le=4, description="Images attached per moment.")] = 2,
        image_px: ImagePx = 768,
        verify: Annotated[
            bool | None,
            Field(
                description="Have Jev 1.14 judge whether each moment's transcript answers the question and whether "
                "each frame shows it, rerank by that, and rescue with a frames-only search when nothing is "
                "supported. Sends excerpts and frames to TypeSafe. Default: DVQA_VERIFY (off)."
            ),
        ] = None,
    ) -> Annotated[CallToolResult, AskResult]:
        """The moments most relevant to a question, as evidence: for each, the transcript around it with
        timestamps and the frames from it as images (in `moments` order). Answer from the evidence and cite
        times. With verify, each moment carries calibrated support probabilities and `evidence.supported`
        says whether the video answers the question at all."""
        verify_on = settings.verify_default if verify is None else verify
        scope = app.scope(video_ids)
        hits, raw = await retrieval.search_with_evidence(
            app.store, app.embedder, question, video_ids=scope, top_k=max(12, max_moments * 4)
        )
        candidates = max_moments + 2 if verify_on else max_moments
        moments = [
            m.to_dict()
            for m in retrieval.build_moments(
                app.store, hits, max_moments=candidates, frames_per_moment=frames_per_moment
            )
        ]
        evidence = {**raw.to_dict(), "verified": False}
        if verify_on:
            try:
                moments, verdict = await _verify(question, moments, scope, frames_per_moment)
                evidence.update(verdict)
            except VideoQAError as exc:
                if exc.code == "verify_unavailable":
                    raise
                evidence["verify_error"] = str(exc)
        moments = moments[:max_moments]
        if not moments:
            guidance = "Nothing relevant was retrieved. Check video_list, or rephrase the question."
        elif evidence.get("supported") is False:
            guidance = (
                "The verifier found no moment that answers this. Say the video does not appear to cover it, "
                "unless the images show otherwise."
            )
        else:
            guidance = "Answer only from this evidence and cite times as m:ss. Images follow in moment order."
        result = AskResult(
            question=question,
            moments=[{**m, "title": app.title(m["video_id"])} for m in moments],
            evidence=evidence,
            guidance=guidance,
        )
        images = [(f"Frame from {m['video_id']} at {f['at']}", Path(f["path"])) for m in moments for f in m["frames"]]
        return _with_images(result, await app.images(images, image_px))

    async def _verify(
        question: str, moments: list[dict], scope: list[str] | None, frames_per_moment: int
    ) -> tuple[list[dict], dict]:
        """Judge every moment, rerank by support, and fall back to frames when nothing is supported."""
        verifier = app.verifier

        async def judge(moment: dict) -> dict:
            paths = [f["path"] for f in moment["frames"] if f.get("path")]
            verdict = await verifier.judge(question, moment["transcript"], paths)
            for frame in moment["frames"]:
                frame["shows"] = round(verdict.frames.get(frame["path"], 0.0), 3) if frame.get("path") else None
            moment["transcript_answers"] = None if verdict.answers is None else round(verdict.answers, 3)
            moment["support"] = round(verdict.support, 3)
            return moment

        try:
            moments = list(await asyncio.gather(*(judge(m) for m in moments)))
        except VideoQAError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors become a soft failure
            raise VideoQAError("verify_failed", f"{type(exc).__name__}: {exc}") from exc
        moments.sort(key=lambda m: -(m["support"] or 0.0))
        rescued = False
        if not moments or moments[0]["support"] < SUPPORT_THRESHOLD:
            # Some answers are only on screen. Judge the best frame matches one by one, then build moments
            # around the frames that pass, so a good frame is never hidden behind a merged window.
            frame_hits, _ = await retrieval.search_with_evidence(
                app.store, app.embedder, question, video_ids=scope, modalities=["frames"], top_k=6
            )
            judged = {f["path"]: f.get("shows") for m in moments for f in m["frames"] if f.get("path")}
            fresh = [h for h in frame_hits if h.frame_path and h.frame_path not in judged]
            try:
                scores = await asyncio.gather(*(verifier.frame_shows(question, h.frame_path) for h in fresh))
                judged.update({h.frame_path: s for h, s in zip(fresh, scores, strict=True)})
                good = [h for h in frame_hits if (judged.get(h.frame_path) or 0.0) >= SUPPORT_THRESHOLD]
                extra = [
                    m.to_dict()
                    for m in retrieval.build_moments(app.store, good, frames_per_moment=max(1, frames_per_moment))
                ]
                answers = await asyncio.gather(*(verifier.transcript_answers(question, m["transcript"]) for m in extra))
            except VideoQAError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VideoQAError("verify_failed", f"{type(exc).__name__}: {exc}") from exc
            for moment, answer in zip(extra, answers, strict=True):
                for frame in moment["frames"]:
                    frame["shows"] = round(judged.get(frame["path"]) or 0.0, 3)
                moment["transcript_answers"] = None if answer is None else round(answer, 3)
                moment["support"] = round(max([answer or 0.0, *(f["shows"] for f in moment["frames"])]), 3)
            if extra:
                rescued = True
                moments = sorted(extra, key=lambda m: -m["support"]) + moments
        best = max((m["support"] for m in moments), default=0.0)
        return moments, {
            "verified": True,
            "verifier": verifier.model,
            "supported": best >= SUPPORT_THRESHOLD,
            "best_support": round(best, 3),
            "rescued_by_frames": rescued,
        }

    @mcp.tool(title="Search inside videos", annotations=READ)
    async def video_search(
        query: Annotated[str, Field(min_length=1, description="What to look for, in natural language.")],
        video_ids: VideoIds = None,
        modalities: Annotated[
            list[Literal["transcript", "frames"]], Field(description="Search what was said, what was shown, or both.")
        ] = ["transcript", "frames"],  # noqa: B006
        top_k: Annotated[int, Field(ge=1, le=50, description="Hits to return.")] = 10,
        start: Annotated[float | None, Field(ge=0, description="Only after this second.")] = None,
        end: Annotated[float | None, Field(ge=0, description="Only before this second.")] = None,
    ) -> SearchResult:
        """Ranked hits from three channels fused by weighted reciprocal rank: meaning-based transcript search,
        keyword (BM25) search, and text-to-image search over frames. 'The slide with the pricing table' finds the
        frame even if nobody said it. Each hit lists the channels that matched it, with rank and raw score."""
        hits = await retrieval.search(
            app.store, app.embedder, query,
            video_ids=app.scope(video_ids), modalities=modalities, top_k=top_k, start=start, end=end,
        )  # fmt: skip
        return SearchResult(query=query, count=len(hits), hits=[hit.to_dict() for hit in hits])

    @mcp.tool(title="Find frames that look like an image", annotations=READ)
    async def video_search_by_image(
        image_path: Annotated[str, Field(description="Absolute path to a picture (jpg, png, webp).")],
        video_ids: VideoIds = None,
        top_k: Annotated[int, Field(ge=1, le=50, description="Hits to return.")] = 8,
    ) -> SearchResult:
        """Where a picture, logo, chart or slide appears: frames that look like it, and passages about it."""
        path = settings.check_path(image_path)
        hits = await retrieval.search_by_image(
            app.store, app.embedder, path, video_ids=app.scope(video_ids), top_k=top_k
        )
        return SearchResult(query=str(path), count=len(hits), hits=[hit.to_dict() for hit in hits])

    @mcp.tool(title="Find every time a phrase is said", annotations=READ)
    async def video_find_phrase(
        phrase: Annotated[str, Field(min_length=2, max_length=200, description="Exact words, case-insensitive.")],
        video_ids: VideoIds = None,
        limit: Annotated[int, Field(ge=1, le=500, description="Most occurrences to return.")] = 100,
    ) -> PhraseResult:
        """Every transcript segment containing the exact phrase, in time order. Use it to count mentions or
        collect every place a name or number comes up; use video_search for meaning rather than wording.
        A phrase split across two segments is not matched."""
        rows = app.store.find_phrase(phrase, video_ids=app.scope(video_ids), limit=limit + 1)
        hits = [{**row, "at": retrieval.clock(row["start"])} for row in rows[:limit]]
        return PhraseResult(phrase=phrase, count=len(hits), truncated=len(rows) > limit, hits=hits)

    # ================================================================ inspect

    @mcp.tool(title="Read the transcript", annotations=READ)
    async def video_transcript(
        video_id: VideoId,
        start: Annotated[float | None, Field(ge=0, description="From this second.")] = None,
        end: Annotated[float | None, Field(ge=0, description="Up to this second.")] = None,
        format: Annotated[
            Literal["timestamped", "plain", "srt", "vtt", "json"], Field(description="Output layout.")
        ] = "timestamped",
        max_chars: Annotated[int, Field(ge=500, le=400_000, description="Truncate beyond this.")] = 60_000,
    ) -> TranscriptResult:
        """What was said, for the whole video or a range: timestamped lines, plain text, SRT, WebVTT, or JSON
        segments. Page through long videos with start and end."""
        video = app.video(video_id)
        segments = app.store.segments(video["id"], start, end)
        base = {
            "video_id": video["id"],
            "title": video["title"],
            "language": video["language"],
            "segment_count": len(segments),
            "format": format,
        }
        note = None if segments else "no transcript in this range; the video may be silent or indexed frames-only"
        if format == "json":
            kept, used = [], 0
            for seg in segments:
                used += len(seg["text"]) + 40
                if used > max_chars:
                    break
                kept.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            truncated = len(kept) < len(segments)
            return TranscriptResult(**base, segments=kept, truncated=truncated, note=note)
        text = render_transcript(segments, format)
        truncated = len(text) > max_chars
        if truncated:
            note = "truncated; pass start/end to page through the rest"
        return TranscriptResult(**base, text=text[:max_chars], truncated=truncated, note=note)

    @mcp.tool(title="Look at frames", annotations=READ)
    async def video_frames(
        video_id: VideoId,
        timestamps: Annotated[list[float] | None, Field(max_length=12, description="Exact seconds, up to 12.")] = None,
        start: Annotated[float | None, Field(ge=0, description="Or: evenly spaced frames from this second...")] = None,
        end: Annotated[float | None, Field(ge=0, description="...to this second.")] = None,
        count: Annotated[int, Field(ge=1, le=12, description="Frames across start..end.")] = 6,
        exact: Annotated[bool, Field(description="Decode at the exact instant from the source file.")] = True,
        image_px: ImagePx = 1024,
    ) -> Annotated[CallToolResult, FramesResult]:
        """Frames as images, at given timestamps or spread across a range. With exact=true each is decoded from
        the source at that instant and at higher resolution than the indexed samples, for reading small text."""
        video = app.video(video_id)
        duration = float(video.get("duration") or 0.0)
        if timestamps:
            times = [min(max(0.0, t), max(duration - 0.05, 0.0)) for t in timestamps]
        else:
            lo, hi = start or 0.0, min(end if end is not None else duration, duration)
            if hi <= lo:
                raise VideoQAError("bad_range", f"end must be after start and within the video ({duration:.1f}s)")
            times = [lo + (hi - lo) * (i + 0.5) / count for i in range(count)]
        refs: list[dict] = []
        sources: list[tuple[str, Path | bytes]] = []
        if exact:
            source = app.source_file(video)
            gate = asyncio.Semaphore(settings.cpu_workers)

            async def grab(t: float) -> bytes:
                async with gate:
                    return await media.frame_at(source, t, width=max(image_px, 1280))

            for t, data in zip(times, await asyncio.gather(*(grab(t) for t in times)), strict=True):
                refs.append({"at": retrieval.clock(t), "t": round(t, 3), "path": None})
                sources.append((f"Frame at {retrieval.clock(t)} ({t:.2f}s)", data))
        else:
            seen: set[str] = set()
            for t in times:
                frame = app.store.nearest_frame(video["id"], t)
                if frame and frame["path"] not in seen:
                    seen.add(frame["path"])
                    refs.append({"at": retrieval.clock(frame["t"]), "t": frame["t"], "path": frame["path"]})
                    sources.append((f"Sampled frame at {retrieval.clock(frame['t'])}", Path(frame["path"])))
        result = FramesResult(video_id=video["id"], title=video["title"], exact=exact, frames=refs)
        return _with_images(result, await app.images(sources, image_px))

    @mcp.tool(title="Outline a video", annotations=READ)
    async def video_chapters(
        video_id: VideoId,
        include_images: Annotated[bool, Field(description="Attach one frame per chapter.")] = False,
        image_px: Annotated[int, Field(ge=256, le=1024, description="Long side of chapter images.")] = 512,
    ) -> Annotated[CallToolResult, ChaptersResult]:
        """Chapters with spans, opening words and distinguishing keywords. Uses the uploader's chapters when the
        source has them; otherwise finds topic shifts in the transcript embeddings (frames for silent video)."""
        video = app.video(video_id)
        found = retrieval.chapters(app.store, app.embedder, video)
        result = ChaptersResult(video_id=video["id"], title=video["title"], chapters=found)
        images: list[tuple[str, Path | bytes]] = []
        if include_images:
            images = [
                (f"Chapter {c['index']} at {retrieval.clock(c['start'])}", Path(c["frame_path"]))
                for c in found[:12]
                if c.get("frame_path")
            ]
        return _with_images(result, await app.images(images, image_px))

    @mcp.tool(title="Export a clip", annotations=WRITE)
    async def video_clip(
        video_id: VideoId,
        start: Annotated[float, Field(ge=0, description="Clip start in seconds.")],
        end: Annotated[float, Field(gt=0, description="Clip end in seconds.")],
        exact: Annotated[bool, Field(description="Re-encode for frame-accurate cuts; off snaps to keyframes.")] = False,
        output_dir: OutputDir = None,
    ) -> ClipResult:
        """Cut a time range into its own MP4 with audio. Stream copy is instant; exact re-encodes."""
        video = app.video(video_id)
        if end <= start:
            raise VideoQAError("bad_range", "end must be after start")
        if end - start > 3600:
            raise VideoQAError("bad_range", "clips are limited to one hour")
        source = app.source_file(video)
        dest = app.output_dir(output_dir, settings.downloads_dir / "clips")
        dest = dest / f"{video['id']}_{int(start * 1000)}_{int(end * 1000)}.mp4"
        await media.cut_clip(source, start, end, dest, reencode=exact)
        info = await media.probe(dest)
        return ClipResult(
            path=str(dest), duration=round(info.duration, 2), size_bytes=info.size_bytes,
            has_audio=info.has_audio, exact=exact,
        )  # fmt: skip

    # ================================================================ library

    @mcp.tool(title="List indexed videos", annotations=READ)
    async def video_list() -> VideoList:
        """Every video in the index with its id, title, length, status and index counts."""
        videos = [video_summary(video) for video in app.store.list_videos()]
        return VideoList(count=len(videos), videos=videos)

    @mcp.tool(title="Describe one video", annotations=READ)
    async def video_info(video_id: VideoId) -> VideoDetail:
        """Source, dimensions, language, index counts, completed stages and the uploader's description."""
        return app.video_detail(video_id)

    @mcp.tool(title="Remove a video from the index", annotations=DESTROY)
    async def video_delete(
        video_id: VideoId,
        delete_files: Annotated[bool, Field(description="Also delete the server's downloaded copy and frames.")] = True,
    ) -> DeleteResult:
        """Delete a video's transcript, frames and vectors. A file you ingested by path is never deleted; only
        the server's own downloaded copy and extracted frames are."""
        video = app.video(video_id)
        removed = app.pipeline.delete(video["id"], delete_files=delete_files)
        return DeleteResult(video_id=video["id"], removed=removed, files_deleted=delete_files)

    register_resources(mcp, app)
    register_prompts(mcp)
    return mcp, app
