from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from distributed_video_qa_mcp import download, media, retrieval
from distributed_video_qa_mcp.config import Settings
from distributed_video_qa_mcp.errors import VideoQAError
from distributed_video_qa_mcp.netguard import check_url
from distributed_video_qa_mcp.server import render_transcript
from distributed_video_qa_mcp.store import fts_query
from distributed_video_qa_mcp.transcribe import Segment, build_passages, tidy


def test_plan_chunks_cuts_on_nearby_silence():
    chunks = media.plan_chunks(1500.0, [590.0, 1210.0], target=600.0)
    assert chunks == [(0.0, 590.0), (590.0, 1210.0), (1210.0, 1500.0)]
    assert media.plan_chunks(620.0, [], target=600.0) == [(0.0, 620.0)]


def test_plan_chunks_covers_the_whole_timeline_without_silence():
    chunks = media.plan_chunks(2000.0, [], target=600.0)
    assert chunks[0][0] == 0.0 and chunks[-1][1] == 2000.0
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))


def test_plan_segments_is_contiguous():
    segments = media.plan_segments(100.0, 4)
    assert len(segments) == 4 and segments[0][0] == 0.0 and segments[-1][1] == 100.0
    assert media.plan_segments(10.0, 8) == [(0.0, 10.0)]
    assert media.plan_segments(0.0, 4) == []


def test_dedupe_keeps_changes_and_refreshes_static_shots():
    frames = [media.Frame(t, Path(f"{t}.jpg"), phash=h) for t, h in [(0, 0), (5, 1), (10, 0), (40, 0), (45, 0xFFFF)]]
    kept, dropped = media.dedupe(frames, distance=4, max_gap=30.0)
    assert [f.t for f in kept] == [0, 40, 45]
    assert [f.t for f in dropped] == [5, 10]


def test_thin_subsamples_evenly():
    frames = [media.Frame(float(i), Path(f"{i}.jpg")) for i in range(100)]
    kept, dropped = media.thin(frames, 10)
    assert len(kept) == 10 and len(dropped) == 90 and kept[0].t == 0.0


def test_tidy_merges_repeats_and_sorts():
    out = tidy(
        [Segment(5, 6, "Thanks."), Segment(0, 2, "Hello"), Segment(6.5, 7, "thanks."), Segment(30, 31, "Thanks.")]
    )
    assert [(s.start, s.end, s.text) for s in out] == [(0, 2, "Hello"), (5, 7, "Thanks."), (30, 31, "Thanks.")]


def test_build_passages_overlap_and_coverage():
    segments = [Segment(i * 5.0, i * 5.0 + 5.0, f"sentence {i}") for i in range(20)]
    passages = build_passages(segments, target_seconds=30.0)
    assert passages[0].start == 0.0 and passages[-1].end == 100.0
    assert all(a.end > b.start for a, b in zip(passages, passages[1:], strict=False))
    assert build_passages([]) == []


def test_fts_query_is_quoted_and_safe():
    assert fts_query('what about "NEAR" AND (x)?') == '"what" OR "about" OR "near" OR "and"'
    assert fts_query("!!!") == ""


def test_format_selector():
    assert download.format_selector("video+audio", 720, "mp4", None)[0].startswith("bestvideo*[height<=720]+bestaudio")
    assert download.format_selector("video+audio", None, "mp4", "136") == (
        "136+bestaudio/136",
        ["res", "vcodec:h264", "acodec:aac", "ext:mp4:m4a"],
    )
    assert download.format_selector("video+audio", None, "mkv", "136+140")[0] == "136+140"
    assert download.format_selector("audio_only", None, "mp4", None)[0] == "bestaudio/best"
    with pytest.raises(VideoQAError):
        download.format_selector("video+audio", None, "mp4", "136; rm -rf")
    assert download.parse_quality("720p") == 720 and download.parse_quality("best") is None
    with pytest.raises(VideoQAError):
        download.parse_quality("huge")


def test_summarize_formats_flags_silent_qualities():
    info = {
        "id": "x",
        "title": "t",
        "formats": [
            {"format_id": "18", "vcodec": "avc1.4", "acodec": "mp4a.40", "height": 360, "ext": "mp4", "filesize": 100},
            {"format_id": "136", "vcodec": "avc1.6", "acodec": "none", "height": 720, "ext": "mp4", "filesize": 900},
            {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40", "ext": "m4a", "tbr": 128, "filesize": 50},
            {"format_id": "sb0", "vcodec": "none", "acodec": "none", "ext": "mhtml"},
        ],
    }
    menu = {q["quality"]: q for q in download.summarize_formats(info)["qualities"]}
    assert menu["720p"]["needs_audio_merge"] and menu["720p"]["audio_format_id"] == "140"
    assert menu["720p"]["merged_size_bytes"] == 950
    assert menu["360p"]["has_sound_natively"] and menu["360p"]["audio_format_id"] is None


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://127.0.0.1:8080/x", "http://169.254.169.254/latest", "http://[::1]/", "ftp://a/b",
            "http://user:pw@example.com/", "http://10.0.0.5/v.mp4"]
)  # fmt: skip
def test_netguard_blocks_non_public_targets(url):
    with pytest.raises(VideoQAError):
        check_url(url)


def test_netguard_allows_private_when_opted_in():
    assert check_url("http://127.0.0.1:8080/x", allow_private=True)


def test_check_path_enforces_roots(tmp_path):
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path / "ok",))
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "a.mp4").write_bytes(b"x")
    assert settings.check_path(tmp_path / "ok" / "a.mp4").name == "a.mp4"
    with pytest.raises(VideoQAError):
        settings.check_path(tmp_path / "ok" / ".." / "secret.mp4", must_exist=False)
    with pytest.raises(VideoQAError):
        settings.check_path("/etc/hosts")


def test_boundaries_finds_a_topic_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    vectors = np.vstack(
        [a + 0.05 * rng.normal(size=64) for _ in range(10)] + [b + 0.05 * rng.normal(size=64) for _ in range(10)]
    )
    times = np.arange(20, dtype=np.float32) * 30.0
    assert retrieval.boundaries(times, vectors.astype(np.float32)) == [300.0]


def test_render_transcript_formats():
    segments = [{"start": 1.5, "end": 3.25, "text": "hello"}, {"start": 3661.0, "end": 3662.0, "text": "bye"}]
    assert render_transcript(segments, "srt").startswith("1\n00:00:01,500 --> 00:00:03,250\nhello\n")
    assert "01:01:01.000 --> 01:01:02.000" in render_transcript(segments, "vtt")
    assert render_transcript(segments, "timestamped") == "[0:01] hello\n[1:01:01] bye"
    assert retrieval.clock(59.9) == "0:59"


async def test_token_budget_waits_for_the_window(monkeypatch):
    import asyncio

    from distributed_video_qa_mcp import embed

    monkeypatch.setattr(embed.TokenBudget, "WINDOW", 0.3)
    budget = embed.TokenBudget(1000)
    first = await budget.reserve(800)
    budget.settle(first, 800)
    started = asyncio.get_running_loop().time()
    await budget.reserve(500)
    assert asyncio.get_running_loop().time() - started >= 0.2
    assert 1300 < budget.per_item["image"] < 1700
    before = budget.per_item["image"]
    budget.learn("image", 10, 5000)
    assert budget.per_item["image"] == 0.7 * before + 0.3 * 500
