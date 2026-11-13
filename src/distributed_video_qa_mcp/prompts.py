"""MCP prompts: ready-made workflows that compose the tools.

Each prompt is a short, explicit plan for the model: which tools to call, in
what order, and what the answer must contain. They show up as slash commands
in most MCP hosts.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

CITE = "Cite every claim with its timestamp as m:ss. Say plainly when the video does not cover something."


def register_prompts(mcp: MCPServer) -> None:
    @mcp.prompt(name="video_brief", title="Brief me on a video")
    def video_brief(source: str) -> str:
        """Ingest a video if needed and produce a structured brief: thesis, chapters, key claims, open questions."""
        return f"""Brief me on this video: {source}

1. If it is not in video_list yet, call video_ingest(source) and poll video_job until it succeeds.
2. Call video_chapters(video_id) for the structure, then video_transcript for any chapter you need verbatim.
3. Use video_ask for anything the transcript leaves unclear, and video_frames to read slides or charts.

Write: a one-paragraph summary of the central point; the chapters with spans and one line each; the five most \
important claims; and what the video leaves unanswered. {CITE}"""

    @mcp.prompt(name="video_fact_check", title="Fact-check a video's claims")
    def video_fact_check(video_id: str, topic: str = "") -> str:
        """List the checkable claims a video makes, each with its exact wording, timestamp and on-screen support."""
        focus = f" about {topic}" if topic else ""
        return f"""Fact-check the claims{focus} in video {video_id}.

1. Read the transcript with video_transcript (page with start/end if it is long).
2. For each factual, numeric or causal claim, quote it exactly and record its timestamp. Use video_find_phrase \
to find every repetition of key numbers or names.
3. For claims backed by a chart, table or screenshot, look at it with video_frames(exact=true) and transcribe \
what it actually shows.

Return a table: claim (verbatim), time, type (number / causal / prediction / opinion), what the video offers as \
support, and whether that support matches the claim. Do not judge truth beyond what the video shows unless I \
ask. {CITE}"""

    @mcp.prompt(name="video_study_notes", title="Study notes from a video")
    def video_study_notes(video_id: str) -> str:
        """Turn a lecture or tutorial into structured notes with definitions, steps and the slides that matter."""
        return f"""Make study notes from video {video_id}.

Use video_chapters for structure, video_transcript for wording, and video_frames(exact=true) at each chapter \
start to read slides, code or diagrams on screen. Produce: key terms with definitions, each procedure as \
numbered steps, formulas or code exactly as shown, and five review questions with answers. {CITE}"""

    @mcp.prompt(name="video_compare", title="Compare videos")
    def video_compare(video_ids: str, question: str) -> str:
        """Answer one question across several videos and show where they agree and disagree."""
        return f"""Compare these videos on the question: {question}
Video ids: {video_ids}

Call video_ask with all the ids, then again with each id alone if one dominates the results. For each video, \
state its answer with evidence; then list agreements, contradictions, and what only one of them covers. {CITE}"""

    @mcp.prompt(name="video_find_moment", title="Find a moment")
    def video_find_moment(description: str) -> str:
        """Locate a moment by what was said or shown and verify it visually."""
        return f"""Find the moment: {description}

Call video_search with both modalities. Verify the top two hits with video_frames at their timestamps (exact) \
and the transcript around them. Report the best match as video id and m:ss, what is said and shown there, and \
how confident you are. Offer video_clip if I want it exported."""
