"""Hybrid retrieval over what was said and what was shown, plus moment and chapter building."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embed import QUERY, Embedder
from .store import Store

RRF_K = 60
# The transcript is searched two ways and frames one way. Halving each
# transcript channel keeps "what was said" and "what was shown" at equal weight,
# so a strong frame match is not outvoted just because there are two text channels.
WEIGHTS = {"said_semantic": 0.5, "said_keyword": 0.5, "shown": 1.0, "looks_like": 1.0, "talks_about": 1.0}
MODALITIES = ("transcript", "frames")


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass
class Hit:
    kind: str  # "transcript" | "frame"
    video_id: str
    start: float
    end: float
    score: float
    text: str | None = None
    frame_path: str | None = None
    channels: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "video_id": self.video_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "at": clock(self.start),
            "score": round(self.score, 5),
            "matched_by": {
                name: {"rank": int(v["rank"]), "score": round(v["score"], 4)} for name, v in self.channels.items()
            },
        }
        if self.text is not None:
            out["text"] = self.text
        if self.frame_path is not None:
            out["frame_path"] = self.frame_path
        return out


def _fuse(store: Store, channels: dict[str, tuple[str, list[tuple[int, float]]]], top_k: int) -> list[Hit]:
    """Reciprocal rank fusion: rank position counts, raw scores do not.

    Text-to-text and text-to-image similarities live on different scales, so
    summing them would let one channel drown the other. Ranks are comparable.
    """
    fused: dict[tuple[str, int], float] = {}
    detail: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for name, (kind, results) in channels.items():
        for rank, (ref, score) in enumerate(results, start=1):
            key = (kind, ref)
            fused[key] = fused.get(key, 0.0) + WEIGHTS.get(name, 1.0) / (RRF_K + rank)
            detail.setdefault(key, {})[name] = {"rank": rank, "score": score}
    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    passages = store.passages_by_id([ref for (kind, ref), _ in ordered if kind == "passage"])
    frames = store.frames_by_id([ref for (kind, ref), _ in ordered if kind == "frame"])
    hits: list[Hit] = []
    for (kind, ref), score in ordered:
        if kind == "passage" and ref in passages:
            row = passages[ref]
            hits.append(
                Hit("transcript", row["video_id"], row["start"], row["end"], score, text=row["text"],
                    channels=detail[(kind, ref)])
            )  # fmt: skip
        elif kind == "frame" and ref in frames:
            row = frames[ref]
            hits.append(
                Hit("frame", row["video_id"], row["t"], row["t"], score, frame_path=row["path"],
                    channels=detail[(kind, ref)])
            )  # fmt: skip
    return hits


async def search(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    video_ids: Sequence[str] | None = None,
    modalities: Sequence[str] = MODALITIES,
    top_k: int = 10,
    start: float | None = None,
    end: float | None = None,
) -> list[Hit]:
    """Find moments by meaning (dense), by wording (BM25) and by what is on screen (text-to-image)."""
    vector = (await embedder.embed_texts([query], QUERY))[0]
    depth = max(top_k * 4, 24)
    scope = {"video_ids": video_ids, "start": start, "end": end, "top_k": depth}
    channels: dict[str, tuple[str, list[tuple[int, float]]]] = {}
    if "transcript" in modalities:
        channels["said_semantic"] = ("passage", store.dense("passage", vector, embedder.model, **scope))
        channels["said_keyword"] = ("passage", store.lexical(query, **scope))
    if "frames" in modalities:
        channels["shown"] = ("frame", store.dense("frame", vector, embedder.model, **scope))
    return _fuse(store, channels, top_k)


async def search_by_image(
    store: Store,
    embedder: Embedder,
    image: Path | bytes,
    *,
    video_ids: Sequence[str] | None = None,
    modalities: Sequence[str] = MODALITIES,
    top_k: int = 10,
) -> list[Hit]:
    """Find frames that look like an image, and passages that talk about what it shows."""
    vector = (await embedder.embed_images([image], QUERY))[0]
    depth = max(top_k * 4, 24)
    channels: dict[str, tuple[str, list[tuple[int, float]]]] = {}
    if "frames" in modalities:
        channels["looks_like"] = (
            "frame", store.dense("frame", vector, embedder.model, video_ids=video_ids, top_k=depth)
        )  # fmt: skip
    if "transcript" in modalities:
        channels["talks_about"] = (
            "passage", store.dense("passage", vector, embedder.model, video_ids=video_ids, top_k=depth)
        )  # fmt: skip
    return _fuse(store, channels, top_k)


@dataclass
class Moment:
    video_id: str
    start: float
    end: float
    score: float
    hits: list[Hit]
    transcript: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "span": f"{clock(self.start)}-{clock(self.end)}",
            "score": round(self.score, 5),
            "matched_by": sorted({name for hit in self.hits for name in hit.channels}),
            "transcript": [
                {"at": clock(seg["start"]), "start": round(seg["start"], 2), "text": seg["text"]}
                for seg in self.transcript
            ],
            "frames": [{"at": clock(f["t"]), "t": round(f["t"], 2), "path": f["path"]} for f in self.frames],
        }


def build_moments(
    store: Store,
    hits: Sequence[Hit],
    *,
    max_moments: int = 5,
    join_gap: float = 20.0,
    max_span: float = 75.0,
    context: float = 8.0,
    frames_per_moment: int = 2,
) -> list[Moment]:
    """Merge nearby hits into time windows and attach the surrounding transcript and frames.

    A passage hit and a frame hit thirty seconds apart are usually the same
    scene. Returning one window with both is more useful than two fragments.
    """
    by_video: dict[str, list[Hit]] = {}
    for hit in hits:
        by_video.setdefault(hit.video_id, []).append(hit)
    moments: list[Moment] = []
    for video_id, group in by_video.items():
        group.sort(key=lambda h: h.start)
        current: Moment | None = None
        for hit in group:
            if current and hit.start - current.end <= join_gap and hit.end - current.start <= max_span:
                current.end = max(current.end, hit.end)
                current.score += hit.score
                current.hits.append(hit)
            else:
                current = Moment(video_id, hit.start, hit.end, hit.score, [hit])
                moments.append(current)
    moments.sort(key=lambda m: -m.score)
    moments = moments[:max_moments]
    for moment in moments:
        lo, hi = max(0.0, moment.start - context), moment.end + context
        moment.transcript = store.segments(moment.video_id, lo, hi)
        matched = sorted((h for h in moment.hits if h.kind == "frame"), key=lambda h: -h.score)
        chosen: dict[str, dict[str, Any]] = {}
        for hit in matched[:frames_per_moment]:
            if hit.frame_path:
                chosen[hit.frame_path] = {"t": hit.start, "path": hit.frame_path}
        if len(chosen) < frames_per_moment:
            # No frame matched on its own, so show what was on screen while it was said.
            nearby = store.frames(moment.video_id, lo, hi)
            middle = (moment.start + moment.end) / 2
            for frame in sorted(nearby, key=lambda f: abs(f["t"] - middle)):
                if len(chosen) >= frames_per_moment:
                    break
                chosen.setdefault(frame["path"], {"t": frame["t"], "path": frame["path"]})
            if not chosen:
                nearest = store.nearest_frame(moment.video_id, middle)
                if nearest and abs(nearest["t"] - middle) <= 60:
                    chosen[nearest["path"]] = {"t": nearest["t"], "path": nearest["path"]}
        moment.frames = sorted(chosen.values(), key=lambda f: f["t"])
    return moments


# ---------------------------------------------------------------- chapters

_STOP = frozenset(
    """a about after again all also am an and any are as at be because been before being but by can could
    did do does doing down for from get go going got had has have he her here him his how i if in into is
    it its just know like me more most my no not now of on one only or other our out over really right
    so some than that the their them then there these they thing things think this those to too up us
    very was we well were what when where which who why will with would yeah you your
    i'm it's that's they're we're you're don't didn't can't won't isn't there's here's let's i've you've
    i'd i'll you'll we'll he's she's what's want say said see look make made new way lot even still much
    many every each another going gonna actually basically""".split()
)
_TOKEN = re.compile(r"[a-z][a-z'-]{2,}")


def _keywords(texts: Sequence[str], background: Counter[str], total_docs: int, limit: int = 6) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(t for t in _TOKEN.findall(text.lower()) if t not in _STOP)
    scored = {term: count * math.log((1 + total_docs) / (1 + background[term])) for term, count in counts.items()}
    return [term for term, _ in sorted(scored.items(), key=lambda item: -item[1])[:limit]]


def boundaries(
    times: np.ndarray, vectors: np.ndarray, *, window: int = 3, min_length: float = 45.0, max_chapters: int = 24
) -> list[float]:
    """Find topic shifts: places where the content just before stops resembling the content just after.

    This is TextTiling on embeddings. For each gap, average `window` vectors on
    either side and take their cosine similarity; deep local dips are boundaries.
    """
    count = len(times)
    if count < 2 * window + 1:
        return []
    similarity = np.ones(count, dtype=np.float32)
    for gap in range(window, count - window + 1):
        left = vectors[gap - window : gap].mean(axis=0)
        right = vectors[gap : gap + window].mean(axis=0)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
        similarity[gap - 1] = float(left @ right) / denom
    inner = similarity[window - 1 : count - window]
    if inner.size == 0:
        return []
    cutoff = float(inner.mean() - 0.5 * inner.std())
    candidates = []
    for i in range(window - 1, count - window):
        value = similarity[i]
        if value < cutoff and value <= similarity[i - 1] and value <= similarity[min(i + 1, count - 1)]:
            candidates.append((float(value), float(times[i + 1])))
    chosen: list[float] = []
    for _, t in sorted(candidates):
        if t - float(times[0]) < min_length or float(times[-1]) - t < min_length / 2:
            continue
        if all(abs(t - other) >= min_length for other in chosen):
            chosen.append(t)
        if len(chosen) >= max_chapters - 1:
            break
    return sorted(chosen)


def chapters(store: Store, embedder: Embedder, video: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a video into chapters from its own content. Uploader chapters win when present."""
    video_id = video["id"]
    duration = float(video.get("duration") or 0.0)
    provided = [c for c in (video.get("meta") or {}).get("chapters") or [] if c.get("title")]
    passages = store.passages(video_id)
    if provided:
        cuts = [float(c.get("start_time") or 0.0) for c in provided]
        titles: list[str | None] = [str(c["title"]) for c in provided]
        source = "uploader"
    else:
        times, vectors = store.vectors_for(video_id, "passage", embedder.model, embedder.dim)
        source = "transcript"
        if len(times) < 7:
            times, vectors = store.vectors_for(video_id, "frame", embedder.model, embedder.dim)
            source = "frames"
        cuts = [0.0, *boundaries(times, vectors)]
        titles = [None] * len(cuts)
    background: Counter[str] = Counter()
    for passage in passages:
        background.update({t for t in _TOKEN.findall(passage["text"].lower()) if t not in _STOP})
    out: list[dict[str, Any]] = []
    for index, start in enumerate(cuts):
        end = cuts[index + 1] if index + 1 < len(cuts) else duration
        inside = [p for p in passages if start <= (p["start"] + p["end"]) / 2 < end]
        segments = store.segments(video_id, start, start + 20)
        opening = " ".join(s["text"] for s in segments if s["start"] >= start - 0.5)[:240]
        frame = store.nearest_frame(video_id, start + min(3.0, max(0.0, (end - start) / 2)))
        out.append(
            {
                "index": index + 1,
                "start": round(start, 2),
                "end": round(end, 2),
                "span": f"{clock(start)}-{clock(end)}",
                "title": titles[index],
                "opens_with": opening,
                "keywords": _keywords([p["text"] for p in inside], background, len(passages)),
                "frame_path": frame["path"] if frame else None,
                "source": source,
            }
        )
    return out
