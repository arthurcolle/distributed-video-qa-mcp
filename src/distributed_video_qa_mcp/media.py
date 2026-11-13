"""ffmpeg/ffprobe work: probing, audio extraction, parallel frame sampling, clips."""

from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .errors import VideoQAError

_PTS = re.compile(rb"pts_time:\s*(-?[0-9.]+)")
_SILENCE_START = re.compile(rb"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(rb"silence_end:\s*(-?[0-9.]+)")


def require_tools() -> dict[str, str]:
    found: dict[str, str] = {}
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if not path:
            raise VideoQAError("missing_tool", f"{tool} is not on PATH; install FFmpeg")
        found[tool] = path
    return found


async def run(cmd: Sequence[str], *, timeout: float | None = None, check: bool = True) -> tuple[int, bytes, bytes]:
    """Run a subprocess without a shell and kill it if the caller is cancelled."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.CancelledError, TimeoutError):
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    code = proc.returncode or 0
    if check and code != 0:
        tail = err.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise VideoQAError("ffmpeg_failed", f"{Path(cmd[0]).name} exited {code}: {' | '.join(tail)}")
    return code, out, err


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    num, _, den = value.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


async def probe(path: Path) -> MediaInfo:
    _, out, _ = await run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    data = json.loads(out or b"{}")
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    durations = [fmt.get("duration")] + [s.get("duration") for s in streams]
    duration = max((float(d) for d in durations if d not in (None, "N/A")), default=0.0)
    if not video and not audio:
        raise VideoQAError("not_media", f"{path.name} has no audio or video stream")
    return MediaInfo(
        duration=duration,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        fps=_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else 0.0,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        size_bytes=int(fmt.get("size") or 0),
    )


# ---------------------------------------------------------------- audio


async def extract_audio(src: Path, dest: Path) -> Path:
    """Write 16 kHz mono FLAC, the format every Whisper backend takes directly."""
    await run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
        + ["-c:a", "flac", str(dest)],
        timeout=3600,
    )
    return dest


async def detect_silences(audio: Path, *, noise_db: int = -32, min_len: float = 0.4) -> list[float]:
    """Midpoints of silent stretches, which are safe places to cut audio."""
    _, _, err = await run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(audio)]
        + ["-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
        timeout=3600,
    )
    starts = [float(m) for m in _SILENCE_START.findall(err)]
    ends = [float(m) for m in _SILENCE_END.findall(err)]
    return [(s + e) / 2 for s, e in zip(starts, ends, strict=False) if e > s]


def plan_chunks(
    duration: float, cut_points: Sequence[float], *, target: float, slack: float = 45.0
) -> list[tuple[float, float]]:
    """Split [0, duration] into pieces near `target` seconds, cutting on silence when one is close."""
    if duration <= target + slack:
        return [(0.0, duration)]
    cuts = sorted(c for c in cut_points if 0 < c < duration)
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > target + slack:
        ideal = start + target
        near = [c for c in cuts if abs(c - ideal) <= slack and c > start + 1.0]
        end = min(near, key=lambda c: abs(c - ideal)) if near else ideal
        chunks.append((start, end))
        start = end
    chunks.append((start, duration))
    return chunks


async def cut_audio(audio: Path, start: float, end: float, dest: Path) -> Path:
    await run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
        + ["-i", str(audio), "-c:a", "flac", str(dest)],
        timeout=1800,
    )
    return dest


# ---------------------------------------------------------------- frames


@dataclass
class Frame:
    t: float
    path: Path
    scene_cut: bool = False
    phash: int = 0
    colour: tuple[int, int, int] = (0, 0, 0)


def plan_segments(duration: float, workers: int, *, min_len: float = 20.0) -> list[tuple[float, float]]:
    """Cut the timeline into contiguous pieces, one ffmpeg process each."""
    if duration <= 0:
        return []
    count = max(1, min(workers, int(duration // min_len) or 1))
    step = duration / count
    return [(i * step, duration if i == count - 1 else (i + 1) * step) for i in range(count)]


async def _extract_segment(
    src: Path,
    out_dir: Path,
    index: int,
    start: float,
    end: float,
    *,
    interval: float,
    scene_threshold: float,
    scene_min_gap: float,
    width: int,
) -> list[Frame]:
    # Keep a frame when it is the first in the segment, when `interval` seconds have
    # passed since the last kept frame, or on a scene cut that is not too close to it.
    select = (
        f"isnan(prev_selected_t)+gte(t-prev_selected_t,{interval:.3f})"
        f"+gt(scene,{scene_threshold:.3f})*gte(t-prev_selected_t,{scene_min_gap:.3f})"
    )
    pattern = out_dir / f"seg{index:03d}_%06d.jpg"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), "-an", "-sn"]
    cmd += ["-vf", f"select='{select}',showinfo,scale='min({width},iw)':-2"]
    cmd += ["-fps_mode", "vfr", "-q:v", "3", str(pattern)]
    _, _, err = await run(cmd, timeout=max(600.0, (end - start) * 4))
    times = [float(m) for m in _PTS.findall(err)]
    files = sorted(out_dir.glob(f"seg{index:03d}_*.jpg"))
    frames: list[Frame] = []
    previous: float | None = None
    for local_t, file in zip(times, files, strict=False):
        cut = previous is not None and (local_t - previous) < interval - 0.05
        frames.append(Frame(t=round(start + max(local_t, 0.0), 3), path=file, scene_cut=cut))
        previous = local_t
    for orphan in files[len(frames) :]:
        orphan.unlink(missing_ok=True)
    return frames


async def extract_frames(
    src: Path,
    out_dir: Path,
    duration: float,
    *,
    interval: float,
    scene_threshold: float,
    scene_min_gap: float,
    width: int,
    workers: int,
    max_frames: int,
) -> list[Frame]:
    """Sample frames on a fixed cadence plus scene cuts, decoding segments in parallel."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # Never plan more interval frames than the budget allows.
    interval = max(interval, duration / max(max_frames, 1))
    segments = plan_segments(duration, workers)
    gate = asyncio.Semaphore(workers)

    async def one(index: int, start: float, end: float) -> list[Frame]:
        async with gate:
            return await _extract_segment(
                src,
                out_dir,
                index,
                start,
                end,
                interval=interval,
                scene_threshold=scene_threshold,
                scene_min_gap=scene_min_gap,
                width=width,
            )

    parts = await asyncio.gather(*(one(i, s, e) for i, (s, e) in enumerate(segments)))
    frames = sorted((f for part in parts for f in part), key=lambda f: f.t)
    return frames


def fingerprint(path: Path, size: int = 8) -> tuple[int, tuple[int, int, int]]:
    """A 64-bit difference hash plus the mean colour.

    The hash captures layout and is blind to a uniform colour change (a cut from
    one flat title card to another), so the mean colour is compared as well.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = rgb.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR).tobytes()
        mean = rgb.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (pixels[base + col] > pixels[base + col + 1])
    return bits, (int(mean[0]), int(mean[1]), int(mean[2]))  # type: ignore[index]


async def hash_frames(frames: list[Frame], workers: int) -> None:
    gate = asyncio.Semaphore(max(1, workers))

    async def one(frame: Frame) -> None:
        async with gate:
            frame.phash, frame.colour = await asyncio.to_thread(fingerprint, frame.path)

    await asyncio.gather(*(one(f) for f in frames))


def _alike(a: Frame, b: Frame, distance: int, colour_tolerance: int) -> bool:
    if (a.phash ^ b.phash).bit_count() > distance:
        return False
    return max(abs(x - y) for x, y in zip(a.colour, b.colour, strict=True)) <= colour_tolerance


def dedupe(
    frames: list[Frame], *, distance: int, max_gap: float, colour_tolerance: int = 12
) -> tuple[list[Frame], list[Frame]]:
    """Drop a frame that looks like the last kept one, unless `max_gap` seconds have passed.

    A static shot (a talking head, a slide left on screen) would otherwise spend
    most of the embedding budget on copies of the same picture.
    """
    kept: list[Frame] = []
    dropped: list[Frame] = []
    for frame in frames:
        last = kept[-1] if kept else None
        if last is not None and _alike(frame, last, distance, colour_tolerance) and frame.t - last.t < max_gap:
            dropped.append(frame)
        else:
            kept.append(frame)
    return kept, dropped


def thin(frames: list[Frame], limit: int) -> tuple[list[Frame], list[Frame]]:
    """Evenly subsample down to `limit` frames."""
    if len(frames) <= limit:
        return frames, []
    step = len(frames) / limit
    keep = {min(len(frames) - 1, math.floor(i * step)) for i in range(limit)}
    kept = [f for i, f in enumerate(frames) if i in keep]
    dropped = [f for i, f in enumerate(frames) if i not in keep]
    return kept, dropped


async def frame_at(src: Path, t: float, *, width: int = 1280) -> bytes:
    """Decode one frame at an exact timestamp and return it as JPEG bytes."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(src)]
    cmd += ["-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", "2"]
    cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    _, out, _ = await run(cmd, timeout=120)
    if not out:
        raise VideoQAError("no_frame", f"no frame could be decoded at {t:.2f}s")
    return out


async def cut_clip(src: Path, start: float, end: float, dest: Path, *, reencode: bool) -> Path:
    """Export [start, end]. Stream copy is instant but snaps to keyframes; re-encode is exact."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src)]
    cmd += ["-t", f"{end - start:.3f}"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd += ["-movflags", "+faststart", str(dest)]
    await run(cmd, timeout=3600)
    return dest
