"""MCP resources: the library and each video's transcript, details and frames, addressable by URI.

Resources let a client attach a transcript or a frame to a conversation directly
(an @-mention in most hosts) without a tool call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .app import App, jpeg, render_transcript, video_summary
from .errors import VideoQAError


def register_resources(mcp: MCPServer, app: App) -> None:
    @mcp.resource(
        "video://library", name="library", title="Indexed videos", mime_type="application/json",
        description="Every indexed video with id, title, length and status.",
    )  # fmt: skip
    def library() -> str:
        videos = [video_summary(v).model_dump(mode="json") for v in app.store.list_videos()]
        return json.dumps({"count": len(videos), "videos": videos}, indent=2)

    @mcp.resource(
        "video://{video_id}/transcript", name="transcript", title="Transcript", mime_type="text/plain",
        description="The full transcript with [m:ss] timestamps.",
    )  # fmt: skip
    def transcript(video_id: str) -> str:
        video = app.video(video_id)
        segments = app.store.segments(video["id"])
        return f"# {video['title']}\n\n" + render_transcript(segments, "timestamped")

    @mcp.resource(
        "video://{video_id}/subtitles.srt", name="subtitles", title="Subtitles (SRT)", mime_type="application/x-subrip",
        description="The transcript as an SRT subtitle file.",
    )  # fmt: skip
    def subtitles(video_id: str) -> str:
        video = app.video(video_id)
        return render_transcript(app.store.segments(video["id"]), "srt")

    @mcp.resource(
        "video://{video_id}/info", name="info", title="Video details", mime_type="application/json",
        description="Source, dimensions, language, index counts and description.",
    )  # fmt: skip
    def info(video_id: str) -> str:
        return app.video_detail(video_id).model_dump_json(indent=2)

    @mcp.resource(
        "video://{video_id}/frame/{seconds}", name="frame", title="Frame at a time", mime_type="image/jpeg",
        description="The indexed frame nearest to a time in seconds, as JPEG.",
    )  # fmt: skip
    async def frame(video_id: str, seconds: str) -> bytes:
        video = app.video(video_id)
        try:
            t = float(seconds)
        except ValueError as exc:
            raise VideoQAError("bad_range", f"{seconds!r} is not a number of seconds") from exc
        nearest = app.store.nearest_frame(video["id"], t)
        if nearest is None:
            raise VideoQAError("no_frame", "this video was indexed without frames")
        return await asyncio.to_thread(jpeg, Path(nearest["path"]), 1280)
