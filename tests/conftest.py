from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from distributed_video_qa_mcp.config import Settings
from distributed_video_qa_mcp.transcribe import Segment

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

SCRIPT = [
    (0.5, 3.5, "Welcome. This first part is about the red planet Mars and its dust storms."),
    (4.0, 7.5, "Rovers measure wind speed on the Martian surface every day."),
    (8.5, 11.5, "Now we move to forests. Green leaves turn sunlight into sugar by photosynthesis."),
    (12.0, 15.5, "Chlorophyll absorbs red and blue light and reflects green."),
    (16.5, 19.5, "Finally the ocean. Blue whales are the largest animals that have ever lived."),
    (20.0, 23.5, "A blue whale eats several tonnes of krill in a single day."),
]


class FakeTranscriber:
    """Stands in for Whisper so tests do not need a model or a network."""

    name = "fake"
    parallel = False

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio: Path, language: str | None):
        self.calls += 1
        assert audio.exists()
        return [Segment(*row) for row in SCRIPT], "en"


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """24 seconds: 8s red, 8s green, 8s blue, with a tone for audio."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    colours = "".join(f"color=c={c}:s=320x240:r=10:d=8[v{i}];" for i, c in enumerate(("red", "green", "blue")))
    graph = colours + "[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=24"]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out


@pytest.fixture
def settings(tmp_path: Path, sample_video: Path) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        embed_backend="hash",
        embed_dim=256,
        frame_interval=2.0,
        cpu_workers=2,
        allowed_roots=(tmp_path, sample_video.parent),
    )
