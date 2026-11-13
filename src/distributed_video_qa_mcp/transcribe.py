"""Speech to timestamped text, on-device when possible and in parallel chunks over an API otherwise."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import platform
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import media
from .config import Settings
from .errors import VideoQAError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Transcript:
    segments: list[Segment]
    language: str | None
    backend: str


class Transcriber(Protocol):
    name: str
    parallel: bool

    async def transcribe(self, audio: Path, language: str | None) -> tuple[list[Segment], str | None]: ...


def _segments(raw: list[dict[str, Any]]) -> list[Segment]:
    out: list[Segment] = []
    for item in raw:
        text = str(item.get("text") or "").strip()
        if text:
            out.append(Segment(float(item.get("start") or 0.0), float(item.get("end") or 0.0), text))
    return out


# One model in memory, one inference at a time: local backends are bound by the
# accelerator, not by Python, so extra threads would only contend for it.
_LOCAL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


class MlxWhisper:
    parallel = False

    def __init__(self, model: str | None) -> None:
        self.model = model or "mlx-community/whisper-large-v3-turbo"
        self.name = f"mlx:{self.model}"

    async def transcribe(self, audio: Path, language: str | None) -> tuple[list[Segment], str | None]:
        def work() -> tuple[list[Segment], str | None]:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                str(audio),
                path_or_hf_repo=self.model,
                language=language,
                verbose=None,
                condition_on_previous_text=False,
            )
            return _segments(result.get("segments", [])), result.get("language")

        return await asyncio.get_running_loop().run_in_executor(_LOCAL, work)


class FasterWhisper:
    parallel = False
    _models: dict[str, Any] = {}

    def __init__(self, model: str | None) -> None:
        self.model = model or "large-v3-turbo"
        self.name = f"faster-whisper:{self.model}"

    async def transcribe(self, audio: Path, language: str | None) -> tuple[list[Segment], str | None]:
        def work() -> tuple[list[Segment], str | None]:
            from faster_whisper import WhisperModel

            if self.model not in self._models:
                self._models[self.model] = WhisperModel(self.model, device="auto", compute_type="auto")
            parts, info = self._models[self.model].transcribe(
                str(audio), language=language, vad_filter=True, condition_on_previous_text=False
            )
            segments = [Segment(float(p.start), float(p.end), p.text.strip()) for p in parts if p.text.strip()]
            return segments, getattr(info, "language", None)

        return await asyncio.get_running_loop().run_in_executor(_LOCAL, work)


class ApiWhisper:
    """Any OpenAI-compatible /audio/transcriptions endpoint (Groq, OpenAI, a local server)."""

    parallel = True

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = f"api:{model}"

    async def transcribe(self, audio: Path, language: str | None) -> tuple[list[Segment], str | None]:
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language
        payload = await asyncio.to_thread(audio.read_bytes)
        last: Exception | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20)) as client:
            for attempt in range(5):
                try:
                    response = await client.post(
                        f"{self.base_url}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=data,
                        files={"file": (audio.name, payload, "audio/flac")},
                    )
                except httpx.TransportError as exc:
                    last = exc
                else:
                    if response.status_code == 200:
                        body = response.json()
                        return _segments(body.get("segments") or []), body.get("language")
                    if response.status_code not in {408, 409, 429} and response.status_code < 500:
                        raise VideoQAError(
                            "transcribe_failed",
                            f"transcription API returned {response.status_code}: {response.text[:300]}",
                        )
                    last = VideoQAError("transcribe_failed", f"transcription API returned {response.status_code}")
                    retry_after = response.headers.get("retry-after", "")
                    if retry_after.replace(".", "", 1).isdigit():
                        await asyncio.sleep(min(float(retry_after), 60))
                        continue
                await asyncio.sleep(min(30, 1.5 * 2**attempt) + random.random())
        raise VideoQAError("transcribe_failed", f"transcription API kept failing: {last}")


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _api_backend(settings: Settings) -> ApiWhisper | None:
    if settings.transcribe_base_url and settings.transcribe_api_key:
        model = settings.whisper_model or "whisper-large-v3-turbo"
        return ApiWhisper(settings.transcribe_base_url, settings.transcribe_api_key, model)
    if os.environ.get("GROQ_API_KEY"):
        model = settings.whisper_model or "whisper-large-v3-turbo"
        return ApiWhisper("https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], model)
    if os.environ.get("OPENAI_API_KEY"):
        model = settings.whisper_model or "whisper-1"
        return ApiWhisper("https://api.openai.com/v1", os.environ["OPENAI_API_KEY"], model)
    return None


def resolve_backend(settings: Settings) -> Transcriber | None:
    """Pick a transcriber. Returns None when transcription is turned off."""
    choice = settings.transcribe_backend
    if choice == "none":
        return None
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if choice == "mlx" or (choice == "auto" and apple_silicon and _has("mlx_whisper")):
        if not _has("mlx_whisper"):
            raise VideoQAError("no_transcriber", "install the 'mlx' extra to use the MLX Whisper backend")
        return MlxWhisper(settings.whisper_model)
    if choice == "faster-whisper" or (choice == "auto" and _has("faster_whisper")):
        if not _has("faster_whisper"):
            raise VideoQAError("no_transcriber", "install the 'faster-whisper' extra to use that backend")
        return FasterWhisper(settings.whisper_model)
    if choice in {"api", "auto"}:
        backend = _api_backend(settings)
        if backend:
            return backend
    if choice not in {"auto", "api", "mlx", "faster-whisper"}:
        raise VideoQAError("config", f"unknown DVQA_TRANSCRIBE_BACKEND {choice!r}")
    raise VideoQAError(
        "no_transcriber",
        "no speech-to-text backend is available: install the 'mlx' or 'faster-whisper' extra, set "
        "GROQ_API_KEY or OPENAI_API_KEY, or set DVQA_TRANSCRIBE_BACKEND=none to index frames only",
    )


def describe_backend(settings: Settings) -> str:
    try:
        backend = resolve_backend(settings)
    except VideoQAError as exc:
        return f"unavailable ({exc.message})"
    return backend.name if backend else "disabled"


def tidy(segments: list[Segment]) -> list[Segment]:
    """Sort, clamp, and drop the back-to-back repeats Whisper produces over silence or music."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        end = max(seg.end, seg.start)
        if out and out[-1].text.casefold() == seg.text.casefold() and seg.start - out[-1].end < 2.0:
            out[-1] = Segment(out[-1].start, end, out[-1].text)
            continue
        out.append(Segment(round(seg.start, 3), round(end, 3), seg.text))
    return out


async def transcribe(
    audio: Path,
    duration: float,
    workdir: Path,
    settings: Settings,
    *,
    language: str | None = None,
    backend: Transcriber | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> Transcript:
    backend = backend or resolve_backend(settings)
    if backend is None:
        return Transcript([], None, "disabled")
    if not backend.parallel:
        if on_progress:
            on_progress(0.05, f"transcribing with {backend.name}")
        segments, detected = await backend.transcribe(audio, language)
        return Transcript(tidy(segments), detected or language, backend.name)

    # Remote backends take bounded uploads, so cut on silence and fan the pieces out.
    silences = await media.detect_silences(audio) if duration > settings.transcribe_chunk_seconds else []
    chunks = media.plan_chunks(duration, silences, target=float(settings.transcribe_chunk_seconds))
    gate = asyncio.Semaphore(settings.transcribe_concurrency)
    finished = 0
    languages: list[str] = []

    async def one(index: int, start: float, end: float) -> list[Segment]:
        nonlocal finished
        async with gate:
            piece = audio if len(chunks) == 1 else workdir / f"chunk_{index:04d}.flac"
            if piece is not audio:
                await media.cut_audio(audio, start, end, piece)
            try:
                segments, detected = await backend.transcribe(piece, language)
            finally:
                if piece is not audio:
                    piece.unlink(missing_ok=True)
        if detected:
            languages.append(detected)
        finished += 1
        if on_progress:
            on_progress(finished / len(chunks), f"transcribed {finished}/{len(chunks)} chunks")
        return [Segment(s.start + start, min(s.end + start, end), s.text) for s in segments]

    parts = await asyncio.gather(*(one(i, s, e) for i, (s, e) in enumerate(chunks)))
    detected = max(set(languages), key=languages.count) if languages else language
    return Transcript(tidy([s for part in parts for s in part]), detected, backend.name)


@dataclass(frozen=True)
class Passage:
    start: float
    end: float
    text: str


def build_passages(
    segments: list[Segment], *, target_seconds: float = 30.0, max_chars: int = 900, overlap: int = 1
) -> list[Passage]:
    """Group segments into retrieval passages of roughly `target_seconds`, overlapping by a segment.

    Single Whisper segments are a few seconds long, too short to carry a claim.
    The overlap keeps a sentence that straddles a boundary findable from both sides.
    """
    passages: list[Passage] = []
    i = 0
    while i < len(segments):
        j = i
        chars = 0
        while j < len(segments):
            chars += len(segments[j].text) + 1
            span = segments[j].end - segments[i].start
            j += 1
            if span >= target_seconds or chars >= max_chars:
                break
        group = segments[i:j]
        passages.append(Passage(group[0].start, group[-1].end, " ".join(s.text for s in group)))
        if j >= len(segments):
            break
        i = max(i + 1, j - overlap)
    return passages
