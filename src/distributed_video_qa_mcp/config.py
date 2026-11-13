from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import VideoQAError


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"DVQA_{name}")
    return value if value not in (None, "") else default


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise VideoQAError("config", f"DVQA_{name} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise VideoQAError("config", f"DVQA_{name} must be between {lo} and {hi}, got {value}")
    return value


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise VideoQAError("config", f"DVQA_{name} must be a number, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise VideoQAError("config", f"DVQA_{name} must be between {lo} and {hi}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_state_dir() -> Path:
    explicit = _env("STATE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "distributed-video-qa-mcp"


@dataclass(frozen=True)
class Settings:
    state_dir: Path

    # Embeddings. One model embeds transcript text and frame images into the same space.
    embed_backend: str = "jina"  # jina | hash (hash is offline and for tests only)
    embed_model: str = "jina-embeddings-v4"
    embed_dim: int = 1024
    embed_base_url: str = "https://api.jina.ai/v1"
    embed_api_key: str | None = None
    embed_concurrency: int = 4
    embed_text_batch: int = 48
    embed_image_batch: int = 12
    embed_image_px: int = 448
    embed_tokens_per_minute: int = 90_000

    # Transcription.
    transcribe_backend: str = "auto"  # auto | mlx | faster-whisper | api | none
    whisper_model: str | None = None
    transcribe_base_url: str | None = None
    transcribe_api_key: str | None = None
    transcribe_concurrency: int = 4
    transcribe_chunk_seconds: int = 600

    # Frames.
    frame_interval: float = 5.0
    scene_threshold: float = 0.30
    scene_min_gap: float = 1.0
    max_frames: int = 600
    frame_width: int = 1280
    dedupe_distance: int = 4
    dedupe_max_gap: float = 30.0

    # Work limits.
    cpu_workers: int = 4
    max_concurrent_jobs: int = 2
    download_fragments: int = 8
    ingest_height: int = 720
    max_duration: int = 6 * 3600
    max_filesize: int = 8 * 1024**3

    # Boundaries.
    allow_private_hosts: bool = False
    allowed_roots: tuple[Path, ...] = field(default_factory=tuple)
    cookies_file: Path | None = None
    http_token: str | None = None

    @property
    def db_path(self) -> Path:
        return self.state_dir / "index.sqlite3"

    @property
    def media_dir(self) -> Path:
        return self.state_dir / "media"

    @property
    def downloads_dir(self) -> Path:
        return self.state_dir / "downloads"

    def video_dir(self, video_id: str) -> Path:
        return self.media_dir / video_id

    def ensure_dirs(self) -> None:
        for path in (self.state_dir, self.media_dir, self.downloads_dir):
            path.mkdir(parents=True, exist_ok=True)
        try:
            self.state_dir.chmod(0o700)
        except OSError:
            pass

    def with_overrides(self, **changes: object) -> Settings:
        return replace(self, **changes)  # type: ignore[arg-type]

    def check_path(self, path: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve a caller-supplied path and require it to sit under an allowed root."""
        resolved = Path(path).expanduser().resolve()
        roots = [root.resolve() for root in self.allowed_roots] + [self.state_dir.resolve()]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise VideoQAError(
                "path_not_allowed",
                f"{resolved} is outside the allowed roots; set DVQA_ALLOWED_ROOTS to widen them",
            )
        if must_exist and not resolved.exists():
            raise VideoQAError("not_found", f"{resolved} does not exist")
        return resolved


def _default_roots(http: bool) -> tuple[Path, ...]:
    raw = _env("ALLOWED_ROOTS")
    if raw:
        return tuple(Path(part).expanduser() for part in raw.split(os.pathsep) if part)
    if http:
        return ()
    return (Path.home(), Path(tempfile.gettempdir()), Path("/tmp"))  # noqa: S108


def load_settings(*, http: bool = False) -> Settings:
    cpu = os.cpu_count() or 4
    cookies = _env("COOKIES_FILE")
    return Settings(
        state_dir=default_state_dir(),
        embed_backend=(_env("EMBED_BACKEND", "jina") or "jina").lower(),
        embed_model=_env("EMBED_MODEL", "jina-embeddings-v4") or "jina-embeddings-v4",
        embed_dim=_int("EMBED_DIM", 1024, 64, 4096),
        embed_base_url=(_env("EMBED_BASE_URL", "https://api.jina.ai/v1") or "").rstrip("/"),
        embed_api_key=_env("EMBED_API_KEY") or os.environ.get("JINA_API_KEY"),
        embed_concurrency=_int("EMBED_CONCURRENCY", 4, 1, 32),
        embed_text_batch=_int("EMBED_TEXT_BATCH", 48, 1, 256),
        embed_image_batch=_int("EMBED_IMAGE_BATCH", 12, 1, 64),
        embed_image_px=_int("EMBED_IMAGE_PX", 448, 128, 2048),
        embed_tokens_per_minute=_int("EMBED_TPM", 90_000, 1_000, 100_000_000),
        transcribe_backend=(_env("TRANSCRIBE_BACKEND", "auto") or "auto").lower(),
        whisper_model=_env("WHISPER_MODEL"),
        transcribe_base_url=_env("TRANSCRIBE_BASE_URL"),
        transcribe_api_key=_env("TRANSCRIBE_API_KEY"),
        transcribe_concurrency=_int("TRANSCRIBE_CONCURRENCY", 4, 1, 32),
        transcribe_chunk_seconds=_int("TRANSCRIBE_CHUNK_SECONDS", 600, 30, 1500),
        frame_interval=_float("FRAME_INTERVAL", 5.0, 0.2, 600.0),
        scene_threshold=_float("SCENE_THRESHOLD", 0.30, 0.01, 1.0),
        scene_min_gap=_float("SCENE_MIN_GAP", 1.0, 0.1, 60.0),
        max_frames=_int("MAX_FRAMES", 600, 1, 20000),
        frame_width=_int("FRAME_WIDTH", 1280, 160, 3840),
        dedupe_distance=_int("DEDUPE_DISTANCE", 4, 0, 32),
        dedupe_max_gap=_float("DEDUPE_MAX_GAP", 30.0, 0.0, 3600.0),
        cpu_workers=_int("CPU_WORKERS", max(2, min(8, cpu // 2)), 1, 64),
        max_concurrent_jobs=_int("MAX_CONCURRENT_JOBS", 2, 1, 32),
        download_fragments=_int("DOWNLOAD_FRAGMENTS", 8, 1, 32),
        ingest_height=_int("INGEST_HEIGHT", 720, 144, 4320),
        max_duration=_int("MAX_DURATION", 6 * 3600, 1, 48 * 3600),
        max_filesize=_int("MAX_FILESIZE", 8 * 1024**3, 1, 1024**4),
        allow_private_hosts=_bool("ALLOW_PRIVATE_HOSTS", False),
        allowed_roots=_default_roots(http),
        cookies_file=Path(cookies).expanduser() if cookies else None,
        http_token=_env("HTTP_TOKEN"),
    )
