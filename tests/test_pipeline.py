from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

from conftest import FakeTranscriber, needs_ffmpeg
from mcp.client import Client

from distributed_video_qa_mcp import media, models
from distributed_video_qa_mcp.errors import CODES
from distributed_video_qa_mcp.server import build_server

pytestmark = needs_ffmpeg

TOOLS = {
    "video_status": models.StatusReport,
    "video_probe": models.ProbeResult,
    "video_download": models.Job,
    "video_ingest": models.Job,
    "video_ingest_batch": models.BatchResult,
    "video_job": models.Job,
    "video_jobs": models.JobList,
    "video_cancel": models.CancelResult,
    "video_ask": models.AskResult,
    "video_search": models.SearchResult,
    "video_search_by_image": models.SearchResult,
    "video_find_phrase": models.PhraseResult,
    "video_transcript": models.TranscriptResult,
    "video_frames": models.FramesResult,
    "video_chapters": models.ChaptersResult,
    "video_clip": models.ClipResult,
    "video_list": models.VideoList,
    "video_info": models.VideoDetail,
    "video_delete": models.DeleteResult,
}


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


async def call(client, tool: str, **args):
    """Call a tool and validate its structured result against the declared model."""
    result = await client.call_tool(tool, args)
    assert not result.is_error, result.content
    return TOOLS[tool].model_validate(result.structured_content), result


async def ingest(client, source: Path, **extra) -> dict:
    job, _ = await call(client, "video_ingest", source=str(source), wait_seconds=120, **extra)
    assert job.status == "succeeded", job
    return job.result


async def test_frames_follow_the_picture(sample_video, tmp_path):
    info = await media.probe(sample_video)
    assert info.has_audio and info.has_video and 23.5 < info.duration < 24.5
    frames = await media.extract_frames(
        sample_video, tmp_path / "f", info.duration,
        interval=2.0, scene_threshold=0.3, scene_min_gap=0.5, width=320, workers=3, max_frames=100,
    )  # fmt: skip
    times = [f.t for f in frames]
    assert times == sorted(times) and times[0] == 0.0 and 12 <= len(frames) <= 16
    await media.hash_frames(frames, 4)
    kept, _ = media.dedupe(frames, distance=4, max_gap=1000.0)
    assert 3 <= len(kept) < len(frames)


async def test_contract(settings):
    async with serve(settings) as (client, _, _):
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == set(TOOLS)
        for name, tool in tools.items():
            assert tool.description and tool.title and tool.annotations, name
            assert tool.output_schema, f"{name} has no output schema"
            declared = set(tool.output_schema.get("properties", {}))
            assert declared == set(TOOLS[name].model_fields), name
        resources = {r.uri for r in (await client.list_resources()).resources}
        templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
        assert "video://library" in resources
        assert {"video://{video_id}/transcript", "video://{video_id}/frame/{seconds}"} <= templates
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert {"video_brief", "video_fact_check", "video_study_notes", "video_compare", "video_find_moment"} == prompts
        status, _ = await call(client, "video_status")
        assert status.embedding.startswith("hash-test-embedder") and status.index.videos == 0


def test_every_raised_error_code_is_registered():
    root = Path(__file__).resolve().parents[1] / "src"
    source = "\n".join(p.read_text() for p in root.rglob("*.py"))
    raised = set(re.findall(r'VideoQAError\(\s*"([a-z_]+)"', source))
    assert raised and raised <= set(CODES), raised - set(CODES)


async def test_ingest_ask_search_and_reuse(settings, sample_video):
    async with serve(settings) as (client, app, fake):
        result = await ingest(client, sample_video)
        video_id = result["video_id"]
        assert result["counts"]["segments"] == 6 and result["counts"]["frames"] >= 3
        assert not (settings.video_dir(video_id) / "audio.flac").exists()

        again = await ingest(client, sample_video)
        assert again["stages"] == {"transcript": "cached", "frames": "cached"} and fake.calls == 1

        listing, _ = await call(client, "video_list")
        assert listing.count == 1 and listing.videos[0].status == "ready"

        hits, _ = await call(client, "video_search", query="blue whale krill", modalities=["transcript"])
        assert hits.hits[0].kind == "transcript" and "whale" in hits.hits[0].text
        assert {"said_semantic", "said_keyword"} <= set(hits.hits[0].matched_by)
        scoped, _ = await call(client, "video_search", query="whale", modalities=["transcript"], end=7)
        assert all(hit.start <= 7 for hit in scoped.hits)

        asked, raw = await call(client, "video_ask", question="what does chlorophyll do", max_moments=2)
        assert any("Chlorophyll" in line.text for m in asked.moments for line in m.transcript)
        assert asked.evidence.keyword_matches >= 1 and asked.evidence.best_said_similarity is not None
        images = [block for block in raw.content if block.type == "image"]
        assert len(images) == sum(len(m.frames) for m in asked.moments) > 0
        assert json.loads(raw.content[0].text) == raw.structured_content

        phrase, _ = await call(client, "video_find_phrase", phrase="blue")
        assert [h.at for h in phrase.hits] == ["0:12", "0:16", "0:20"] and not phrase.truncated

        srt, _ = await call(client, "video_transcript", video_id=video_id, format="srt")
        assert srt.text.startswith("1\n00:00:00,500 --> 00:00:03,500\nWelcome.")
        part, _ = await call(client, "video_transcript", video_id=video_id, start=16, format="json")
        assert [s.start for s in part.segments] == [16.5, 20.0]


async def test_inspect_resources_prompts_and_delete(settings, sample_video, tmp_path):
    async with serve(settings) as (client, app, _):
        video_id = (await ingest(client, sample_video))["video_id"]

        shown, raw = await call(client, "video_frames", video_id=video_id, timestamps=[1.0, 12.0, 20.0])
        assert shown.exact and len(shown.frames) == 3 and all(f.path is None for f in shown.frames)
        assert sum(block.type == "image" for block in raw.content) == 3
        sampled, _ = await call(client, "video_frames", video_id=video_id, start=0, end=24, exact=False)
        assert sampled.frames and all(f.path for f in sampled.frames)

        clip, _ = await call(client, "video_clip", video_id=video_id, start=2, end=6, exact=True,
                             output_dir=str(tmp_path / "clips"))  # fmt: skip
        assert Path(clip.path).exists() and clip.has_audio and 3.5 < clip.duration < 4.6

        chapters, _ = await call(client, "video_chapters", video_id=video_id, include_images=True)
        assert chapters.chapters[0].start == 0.0 and chapters.chapters[-1].end > 23

        image = app.store.frames(video_id)[-1]["path"]
        similar, _ = await call(client, "video_search_by_image", image_path=image, top_k=3)
        assert similar.hits[0].kind == "frame" and similar.hits[0].frame_path == image

        text = (await client.read_resource(f"video://{video_id}/transcript")).contents[0].text
        assert "[0:08] Now we move to forests." in text
        library = json.loads((await client.read_resource("video://library")).contents[0].text)
        assert library["videos"][0]["id"] == video_id
        frame = (await client.read_resource(f"video://{video_id}/frame/13")).contents[0]
        assert frame.mime_type == "image/jpeg" and frame.blob

        brief = await client.get_prompt("video_fact_check", {"video_id": video_id, "topic": "whales"})
        assert "about whales" in brief.messages[0].content.text and video_id in brief.messages[0].content.text

        removed, _ = await call(client, "video_delete", video_id=video_id)
        assert removed.removed and sample_video.exists() and app.store.stats()["vectors"] == 0


async def test_batch_and_errors(settings, sample_video, tmp_path):
    async with serve(settings) as (client, _, fake):
        copy = tmp_path / "copy.mp4"
        copy.write_bytes(sample_video.read_bytes())
        batch, _ = await call(
            client, "video_ingest_batch", sources=[str(sample_video), str(sample_video), "/etc/hosts"],
            transcribe=False, wait_seconds=120,
        )  # fmt: skip
        assert len(batch.jobs) == 1 and batch.jobs[0].status == "succeeded" and fake.calls == 0
        assert len(batch.skipped) == 1 and batch.skipped[0].startswith("/etc/hosts: path_not_allowed")

        async def error(tool: str, **args) -> str:
            """The text after the SDK's 'Error executing tool <name>: ' prefix, which starts with the code."""
            result = await client.call_tool(tool, args)
            assert result.is_error, result
            text = result.content[0].text
            prefix = f"Error executing tool {tool}: "
            assert text.startswith(prefix), text
            return text.removeprefix(prefix)

        assert (await error("video_ingest", source="/etc/hosts")).startswith("path_not_allowed:")
        assert (await error("video_probe", url="http://127.0.0.1:9/video")).startswith("private_host:")
        missing = await error("video_info", video_id="nope")
        assert missing.startswith("not_found:") and "video_list" in missing
        assert (await error("video_ingest", source=str(copy), transcribe=False, frames=False)).startswith(
            "nothing_to_do"
        )

        junk = tmp_path / "notes.mp4"
        junk.write_text("this is not a video")
        job, _ = await call(client, "video_ingest", source=str(junk), wait_seconds=60)
        assert job.status == "failed" and job.error.startswith("ffmpeg_failed:")
