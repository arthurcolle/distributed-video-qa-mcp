from __future__ import annotations

import contextlib
import json
from pathlib import Path

from conftest import FakeTranscriber, needs_ffmpeg
from mcp.client import Client

from distributed_video_qa_mcp import media
from distributed_video_qa_mcp.server import build_server

pytestmark = needs_ffmpeg


@contextlib.asynccontextmanager
async def serve(settings):
    """A server and an in-memory client, opened and closed inside the test's own task."""
    mcp, app = build_server(settings)
    fake = FakeTranscriber()
    app.pipeline._transcriber = fake
    try:
        async with Client(mcp, raise_exceptions=True) as client:
            yield client, app, fake
    finally:
        await app.shutdown()


def data(result) -> dict:
    assert not result.is_error, result.content
    if result.structured_content:
        return result.structured_content
    return json.loads(result.content[0].text)


async def ingest(client, source: Path, **extra) -> dict:
    job = data(await client.call_tool("video_ingest", {"source": str(source), "wait_seconds": 120, **extra}))
    assert job["status"] == "succeeded", job
    return job["result"]


async def test_probe_and_frames_follow_the_picture(sample_video, tmp_path):
    info = await media.probe(sample_video)
    assert info.has_audio and info.has_video and 23.5 < info.duration < 24.5 and (info.width, info.height) == (320, 240)
    frames = await media.extract_frames(
        sample_video, tmp_path / "f", info.duration,
        interval=2.0, scene_threshold=0.3, scene_min_gap=0.5, width=320, workers=3, max_frames=100,
    )  # fmt: skip
    times = [f.t for f in frames]
    assert times == sorted(times) and times[0] == 0.0 and all(f.path.exists() for f in frames)
    assert 12 <= len(frames) <= 16
    await media.hash_frames(frames, 4)
    kept, _ = media.dedupe(frames, distance=4, max_gap=1000.0)
    assert 3 <= len(kept) < len(frames)


async def test_tool_surface(settings):
    async with serve(settings) as served:
        client, _, _ = served
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert {"video_probe", "video_download", "video_ingest", "video_job", "video_ask", "video_search",
                "video_search_by_image", "video_transcript", "video_frames", "video_chapters", "video_clip",
                "video_list", "video_info", "video_delete", "video_status", "video_jobs", "video_cancel"} == set(
                    tools)  # fmt: skip
        assert all(tool.description and tool.annotations for tool in tools.values())
        status = data(await client.call_tool("video_status", {}))
        assert status["embedding"].startswith("hash-test-embedder") and status["index"]["videos"] == 0


async def test_ingest_search_ask_and_reuse(sample_video, settings):
    async with serve(settings) as served:
        client, app, fake = served
        result = await ingest(client, sample_video)
        video_id = result["video_id"]
        assert result["counts"]["segments"] == 6 and result["counts"]["frames"] >= 3
        assert result["counts"]["vectors"] == result["counts"]["passages"] + result["counts"]["frames"]
        assert not (settings.video_dir(video_id) / "audio.flac").exists()

        again = await ingest(client, sample_video)
        assert again["video_id"] == video_id
        assert again["stages"] == {"transcript": "cached", "frames": "cached"} and fake.calls == 1

        listing = data(await client.call_tool("video_list", {}))
        assert listing["count"] == 1 and listing["videos"][0]["status"] == "ready"

        hits = data(await client.call_tool("video_search", {"query": "blue whale krill", "modalities": ["transcript"]}))
        assert hits["hits"][0]["kind"] == "transcript" and "whale" in hits["hits"][0]["text"]
        assert {"said_semantic", "said_keyword"} <= set(hits["hits"][0]["matched_by"])
        scoped = data(
            await client.call_tool("video_search", {"query": "whale", "modalities": ["transcript"], "end": 7})
        )
        assert all(hit["start"] <= 7 for hit in scoped["hits"])

        asked = await client.call_tool("video_ask", {"question": "what does chlorophyll do", "max_moments": 2})
        payload = json.loads(asked.content[0].text)
        assert any("Chlorophyll" in seg["text"] for m in payload["moments"] for seg in m["transcript"])
        images = [block for block in asked.content if block.type == "image"]
        assert images and images[0].mime_type == "image/jpeg"

        srt = data(await client.call_tool("video_transcript", {"video_id": video_id, "format": "srt"}))
        assert srt["text"].startswith("1\n00:00:00,500 --> 00:00:03,500\nWelcome.")
        part = data(await client.call_tool("video_transcript", {"video_id": video_id, "start": 16, "format": "json"}))
        assert [s["start"] for s in part["segments"]] == [16.5, 20.0]


async def test_frames_clip_chapters_and_delete(settings, sample_video, tmp_path):
    async with serve(settings) as served:
        client, app, _ = served
        video_id = (await ingest(client, sample_video))["video_id"]

        shown = await client.call_tool("video_frames", {"video_id": video_id, "timestamps": [1.0, 12.0, 20.0]})
        pictures = [block for block in shown.content if block.type == "image"]
        assert len(pictures) == 3
        sampled = await client.call_tool("video_frames", {"video_id": video_id, "start": 0, "end": 24, "exact": False})
        assert any(block.type == "image" for block in sampled.content)

        clip = data(await client.call_tool("video_clip", {"video_id": video_id, "start": 2, "end": 6, "exact": True,
                                                          "output_dir": str(tmp_path / "clips")}))  # fmt: skip
        assert Path(clip["path"]).exists() and clip["has_audio"] and 3.5 < clip["duration"] < 4.6

        chapters = json.loads((await client.call_tool("video_chapters", {"video_id": video_id})).content[0].text)
        assert chapters["chapters"][0]["start"] == 0.0 and chapters["chapters"][-1]["end"] > 23

        image = app.store.frames(video_id)[-1]["path"]
        similar = data(await client.call_tool("video_search_by_image", {"image_path": image, "top_k": 3}))
        assert similar["hits"][0]["kind"] == "frame" and similar["hits"][0]["frame_path"] == image

        removed = data(await client.call_tool("video_delete", {"video_id": video_id}))
        assert removed["removed"] and sample_video.exists()
        assert data(await client.call_tool("video_list", {}))["count"] == 0
        assert app.store.stats()["vectors"] == 0
        assert data(await client.call_tool("video_search", {"query": "whale"}))["count"] == 0


async def test_frames_only_and_errors(settings, sample_video, tmp_path):
    async with serve(settings) as served:
        client, _, fake = served
        result = await ingest(client, sample_video, transcribe=False)
        assert result["stages"]["transcript"] == "skipped" and result["counts"]["segments"] == 0 and fake.calls == 0

        outside = await client.call_tool("video_ingest", {"source": "/etc/hosts"})
        assert outside.is_error and "path_not_allowed" in outside.content[0].text
        private = await client.call_tool("video_probe", {"url": "http://127.0.0.1:9/video"})
        assert private.is_error and "private_host" in private.content[0].text
        missing = await client.call_tool("video_info", {"video_id": "nope"})
        assert missing.is_error and "not_found" in missing.content[0].text

        junk = tmp_path / "notes.mp4"
        junk.write_text("this is not a video")
        job = data(await client.call_tool("video_ingest", {"source": str(junk), "wait_seconds": 60}))
        assert job["status"] == "failed" and "ffprobe" in job["error"]
