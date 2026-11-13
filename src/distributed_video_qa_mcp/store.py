"""SQLite index: videos, jobs, transcript, frames, vectors and full-text search.

Vectors are float32 blobs. Search loads them once into a NumPy matrix per
(kind, model, dimension) and scores by a single matrix product, which is exact
and stays fast into the hundreds of thousands of vectors on one machine.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  title TEXT,
  uploader TEXT,
  webpage_url TEXT,
  path TEXT,
  duration REAL NOT NULL DEFAULT 0,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  fps REAL NOT NULL DEFAULT 0,
  has_audio INTEGER NOT NULL DEFAULT 0,
  language TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  stages TEXT NOT NULL DEFAULT '{}',
  meta TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  video_id TEXT,
  status TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT '',
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  error TEXT,
  params TEXT NOT NULL DEFAULT '{}',
  result TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  start REAL NOT NULL,
  "end" REAL NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS segments_video ON segments(video_id, start);
CREATE TABLE IF NOT EXISTS passages (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  start REAL NOT NULL,
  "end" REAL NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_video ON passages(video_id, start);
CREATE TABLE IF NOT EXISTS frames (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  t REAL NOT NULL,
  path TEXT NOT NULL,
  scene_cut INTEGER NOT NULL DEFAULT 0,
  phash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS frames_video ON frames(video_id, t);
CREATE TABLE IF NOT EXISTS vectors (
  kind TEXT NOT NULL,
  ref_id INTEGER NOT NULL,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  t0 REAL NOT NULL,
  t1 REAL NOT NULL,
  vec BLOB NOT NULL,
  PRIMARY KEY (kind, ref_id)
);
CREATE INDEX IF NOT EXISTS vectors_video ON vectors(video_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
  text, content='passages', content_rowid='id', tokenize='porter unicode61'
);
"""

_WORD = re.compile(r"[\w']+", re.UNICODE)


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 expression: quoted terms joined by OR."""
    terms = [t for t in dict.fromkeys(w.lower() for w in _WORD.findall(text)) if len(t) > 1]
    return " OR ".join('"' + term.replace('"', "") + '"' for term in terms[:32])


@dataclass
class _VectorCache:
    version: int
    matrix: np.ndarray
    ref_ids: np.ndarray
    video_codes: np.ndarray
    videos: list[str]
    t0: np.ndarray
    t1: np.ndarray


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.executescript(SCHEMA)
        try:
            self._db.executescript(FTS_SCHEMA)
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False
        self._version = 0
        self._cache: dict[tuple[str, str, int], _VectorCache] = {}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ helpers

    def _tx(self) -> _Transaction:
        return _Transaction(self._db, self._lock)

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _row(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, params).fetchone()

    # ------------------------------------------------------------ videos

    @staticmethod
    def _video(row: sqlite3.Row) -> dict[str, Any]:
        video = dict(row)
        video["stages"] = json.loads(video["stages"] or "{}")
        video["meta"] = json.loads(video["meta"] or "{}")
        video["has_audio"] = bool(video["has_audio"])
        return video

    def upsert_video(self, video_id: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        for key in ("stages", "meta"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        with self._tx() as db:
            exists = db.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone()
            if exists:
                if fields:
                    sets = ", ".join(f"{name}=?" for name in fields)
                    db.execute(f"UPDATE videos SET {sets}, updated_at=? WHERE id=?", (*fields.values(), now, video_id))
            else:
                fields.setdefault("source", "")
                fields.setdefault("source_kind", "file")
                names = ", ".join(fields)
                marks = ", ".join("?" for _ in fields)
                db.execute(
                    f"INSERT INTO videos (id, {names}, created_at, updated_at) VALUES (?, {marks}, ?, ?)",
                    (video_id, *fields.values(), now, now),
                )
        video = self.get_video(video_id)
        assert video is not None
        return video

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        row = self._row("SELECT * FROM videos WHERE id=?", (video_id,))
        return self._video(row) if row else None

    def list_videos(self) -> list[dict[str, Any]]:
        rows = self._rows(
            """SELECT v.*,
                 (SELECT COUNT(*) FROM segments s WHERE s.video_id=v.id) AS segment_count,
                 (SELECT COUNT(*) FROM passages p WHERE p.video_id=v.id) AS passage_count,
                 (SELECT COUNT(*) FROM frames f WHERE f.video_id=v.id) AS frame_count
               FROM videos v ORDER BY v.created_at DESC"""
        )
        return [self._video(row) for row in rows]

    def counts(self, video_id: str) -> dict[str, int]:
        row = self._row(
            """SELECT (SELECT COUNT(*) FROM segments WHERE video_id=?1) AS segments,
                      (SELECT COUNT(*) FROM passages WHERE video_id=?1) AS passages,
                      (SELECT COUNT(*) FROM frames WHERE video_id=?1) AS frames,
                      (SELECT COUNT(*) FROM vectors WHERE video_id=?1) AS vectors""",
            (video_id,),
        )
        return dict(row) if row else {}

    def set_stage(self, video_id: str, stage: str, signature: str | None) -> None:
        with self._tx() as db:
            row = db.execute("SELECT stages FROM videos WHERE id=?", (video_id,)).fetchone()
            stages = json.loads(row["stages"] or "{}") if row else {}
            if signature is None:
                stages.pop(stage, None)
            else:
                stages[stage] = signature
            db.execute(
                "UPDATE videos SET stages=?, updated_at=? WHERE id=?",
                (json.dumps(stages), time.time(), video_id),
            )

    def delete_video(self, video_id: str) -> bool:
        with self._tx() as db:
            if self.has_fts:
                db.execute(
                    "INSERT INTO passages_fts(passages_fts, rowid, text) "
                    "SELECT 'delete', id, text FROM passages WHERE video_id=?",
                    (video_id,),
                )
            deleted = db.execute("DELETE FROM videos WHERE id=?", (video_id,)).rowcount > 0
        self._version += 1
        return deleted

    # ------------------------------------------------------------ content

    def replace_transcript(
        self,
        video_id: str,
        segments: Iterable[tuple[float, float, str]],
        passages: Sequence[tuple[float, float, str]],
        vectors: np.ndarray,
        model: str,
    ) -> None:
        with self._tx() as db:
            if self.has_fts:
                db.execute(
                    "INSERT INTO passages_fts(passages_fts, rowid, text) "
                    "SELECT 'delete', id, text FROM passages WHERE video_id=?",
                    (video_id,),
                )
            db.execute("DELETE FROM vectors WHERE video_id=? AND kind='passage'", (video_id,))
            db.execute("DELETE FROM passages WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
            db.executemany(
                'INSERT INTO segments (video_id, idx, start, "end", text) VALUES (?,?,?,?,?)',
                [(video_id, i, s, e, text) for i, (s, e, text) in enumerate(segments)],
            )
            for i, (start, end, text) in enumerate(passages):
                cursor = db.execute(
                    'INSERT INTO passages (video_id, idx, start, "end", text) VALUES (?,?,?,?,?)',
                    (video_id, i, start, end, text),
                )
                ref = cursor.lastrowid
                if self.has_fts:
                    db.execute("INSERT INTO passages_fts(rowid, text) VALUES (?,?)", (ref, text))
                if i < len(vectors):
                    vec = np.ascontiguousarray(vectors[i], dtype=np.float32)
                    db.execute(
                        "INSERT INTO vectors (kind, ref_id, video_id, model, dim, t0, t1, vec) "
                        "VALUES ('passage',?,?,?,?,?,?,?)",
                        (ref, video_id, model, int(vec.shape[0]), start, end, vec.tobytes()),
                    )
        self._version += 1

    def replace_frames(
        self,
        video_id: str,
        frames: Sequence[tuple[float, str, bool, int]],
        vectors: np.ndarray,
        model: str,
    ) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM vectors WHERE video_id=? AND kind='frame'", (video_id,))
            db.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
            for i, (t, path, scene_cut, phash) in enumerate(frames):
                cursor = db.execute(
                    "INSERT INTO frames (video_id, idx, t, path, scene_cut, phash) VALUES (?,?,?,?,?,?)",
                    (video_id, i, t, path, int(scene_cut), f"{phash:016x}"),
                )
                if i < len(vectors):
                    vec = np.ascontiguousarray(vectors[i], dtype=np.float32)
                    db.execute(
                        "INSERT INTO vectors (kind, ref_id, video_id, model, dim, t0, t1, vec) "
                        "VALUES ('frame',?,?,?,?,?,?,?)",
                        (cursor.lastrowid, video_id, model, int(vec.shape[0]), t, t, vec.tobytes()),
                    )
        self._version += 1

    def segments(self, video_id: str, start: float | None = None, end: float | None = None) -> list[dict]:
        sql = 'SELECT idx, start, "end", text FROM segments WHERE video_id=?'
        params: list[Any] = [video_id]
        if start is not None:
            sql += ' AND "end">=?'
            params.append(start)
        if end is not None:
            sql += " AND start<=?"
            params.append(end)
        return [dict(row) for row in self._rows(sql + " ORDER BY start", params)]

    def passages(self, video_id: str) -> list[dict]:
        rows = self._rows(
            'SELECT id, idx, start, "end", text FROM passages WHERE video_id=? ORDER BY start', (video_id,)
        )
        return [dict(row) for row in rows]

    def passages_by_id(self, ids: Sequence[int]) -> dict[int, dict]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self._rows(
            f'SELECT id, video_id, idx, start, "end", text FROM passages WHERE id IN ({marks})', list(ids)
        )
        return {row["id"]: dict(row) for row in rows}

    def frames_by_id(self, ids: Sequence[int]) -> dict[int, dict]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self._rows(f"SELECT id, video_id, idx, t, path, scene_cut FROM frames WHERE id IN ({marks})", list(ids))
        return {row["id"]: dict(row) for row in rows}

    def frames(self, video_id: str, start: float | None = None, end: float | None = None) -> list[dict]:
        sql = "SELECT id, idx, t, path, scene_cut FROM frames WHERE video_id=?"
        params: list[Any] = [video_id]
        if start is not None:
            sql += " AND t>=?"
            params.append(start)
        if end is not None:
            sql += " AND t<=?"
            params.append(end)
        return [dict(row) for row in self._rows(sql + " ORDER BY t", params)]

    def nearest_frame(self, video_id: str, t: float) -> dict | None:
        row = self._row(
            "SELECT id, idx, t, path, scene_cut FROM frames WHERE video_id=? ORDER BY ABS(t-?) LIMIT 1",
            (video_id, t),
        )
        return dict(row) if row else None

    def vectors_for(self, video_id: str, kind: str, model: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """(times, matrix) for one video, ordered by time."""
        rows = self._rows(
            "SELECT t0, vec FROM vectors WHERE video_id=? AND kind=? AND model=? AND dim=? ORDER BY t0",
            (video_id, kind, model, dim),
        )
        if not rows:
            return np.zeros(0, dtype=np.float32), np.zeros((0, dim), dtype=np.float32)
        times = np.asarray([row["t0"] for row in rows], dtype=np.float32)
        matrix = np.frombuffer(b"".join(row["vec"] for row in rows), dtype=np.float32).reshape(len(rows), dim)
        return times, matrix

    # ------------------------------------------------------------ search

    def _vectors(self, kind: str, model: str, dim: int) -> _VectorCache:
        key = (kind, model, dim)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached.version == self._version:
                return cached
            rows = self._db.execute(
                "SELECT ref_id, video_id, t0, t1, vec FROM vectors WHERE kind=? AND model=? AND dim=?",
                (kind, model, dim),
            ).fetchall()
            videos = sorted({row["video_id"] for row in rows})
            code = {video: i for i, video in enumerate(videos)}
            if rows:
                matrix = np.frombuffer(b"".join(row["vec"] for row in rows), dtype=np.float32)
                matrix = matrix.reshape(len(rows), dim)
            else:
                matrix = np.zeros((0, dim), dtype=np.float32)
            cached = _VectorCache(
                version=self._version,
                matrix=matrix,
                ref_ids=np.asarray([row["ref_id"] for row in rows], dtype=np.int64),
                video_codes=np.asarray([code[row["video_id"]] for row in rows], dtype=np.int32),
                videos=videos,
                t0=np.asarray([row["t0"] for row in rows], dtype=np.float32),
                t1=np.asarray([row["t1"] for row in rows], dtype=np.float32),
            )
            self._cache[key] = cached
            return cached

    def dense(
        self,
        kind: str,
        query: np.ndarray,
        model: str,
        *,
        video_ids: Sequence[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Exact cosine top-k over unit vectors. Returns (ref_id, score)."""
        cache = self._vectors(kind, model, int(query.shape[0]))
        if cache.matrix.shape[0] == 0:
            return []
        mask = np.ones(cache.matrix.shape[0], dtype=bool)
        if video_ids is not None:
            wanted = [i for i, video in enumerate(cache.videos) if video in set(video_ids)]
            mask &= np.isin(cache.video_codes, wanted)
        if start is not None:
            mask &= cache.t1 >= start
        if end is not None:
            mask &= cache.t0 <= end
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        scores = cache.matrix[candidates] @ query.astype(np.float32)
        k = min(top_k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(cache.ref_ids[candidates[i]]), float(scores[i])) for i in top]

    def lexical(
        self,
        text: str,
        *,
        video_ids: Sequence[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """BM25 over transcript passages. Returns (passage_id, score), best first."""
        expression = fts_query(text)
        if not self.has_fts or not expression:
            return []
        sql = (
            "SELECT p.id AS id, bm25(passages_fts) AS rank FROM passages_fts "
            "JOIN passages p ON p.id = passages_fts.rowid WHERE passages_fts MATCH ?"
        )
        params: list[Any] = [expression]
        if video_ids is not None:
            if not video_ids:
                return []
            sql += f" AND p.video_id IN ({','.join('?' for _ in video_ids)})"
            params.extend(video_ids)
        if start is not None:
            sql += ' AND p."end">=?'
            params.append(start)
        if end is not None:
            sql += " AND p.start<=?"
            params.append(end)
        params.append(top_k)
        rows = self._rows(sql + " ORDER BY rank LIMIT ?", params)
        return [(row["id"], -float(row["rank"])) for row in rows]

    # ------------------------------------------------------------ jobs

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        job["params"] = json.loads(job["params"] or "{}")
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def create_job(self, job_id: str, kind: str, source: str, params: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._tx() as db:
            db.execute(
                "INSERT INTO jobs (id, kind, source, status, params, created_at, updated_at) "
                "VALUES (?,?,?,'queued',?,?,?)",
                (job_id, kind, source, json.dumps(params), now, now),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def update_job(self, job_id: str, **fields: Any) -> None:
        if "result" in fields and fields["result"] is not None and not isinstance(fields["result"], str):
            fields["result"] = json.dumps(fields["result"])
        sets = ", ".join(f"{name}=?" for name in fields)
        with self._tx() as db:
            db.execute(f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?", (*fields.values(), time.time(), job_id))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._row("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job(row) if row else None

    def list_jobs(self, limit: int = 20, active_only: bool = False) -> list[dict[str, Any]]:
        where = "WHERE status IN ('queued','running')" if active_only else ""
        rows = self._rows(f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._job(row) for row in rows]

    def interrupt_stale_jobs(self) -> int:
        """Jobs left running by a previous process cannot still be running."""
        with self._tx() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='interrupted', message='server restarted; run ingest again to resume', "
                "updated_at=? WHERE status IN ('queued','running')",
                (time.time(),),
            )
            return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        row = self._row(
            """SELECT (SELECT COUNT(*) FROM videos) AS videos,
                      (SELECT COUNT(*) FROM videos WHERE status='ready') AS ready_videos,
                      (SELECT COALESCE(SUM(duration),0) FROM videos WHERE status='ready') AS indexed_seconds,
                      (SELECT COUNT(*) FROM segments) AS segments,
                      (SELECT COUNT(*) FROM passages) AS passages,
                      (SELECT COUNT(*) FROM frames) AS frames,
                      (SELECT COUNT(*) FROM vectors) AS vectors,
                      (SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')) AS active_jobs"""
        )
        models = self._rows("SELECT model, dim, kind, COUNT(*) AS n FROM vectors GROUP BY model, dim, kind")
        out = dict(row) if row else {}
        out["vector_sets"] = [dict(m) for m in models]
        out["full_text_search"] = self.has_fts
        return out


class _Transaction:
    def __init__(self, db: sqlite3.Connection, lock: threading.RLock) -> None:
        self._db = db
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        self._db.execute("BEGIN IMMEDIATE")
        return self._db

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self._db.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self._lock.release()
