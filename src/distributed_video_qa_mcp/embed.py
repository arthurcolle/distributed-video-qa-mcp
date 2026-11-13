"""One embedding space for transcript text and frame images.

Because a multimodal model places a sentence and a picture in the same vector
space, a text question can retrieve the frame that shows the answer without any
captioning step in between.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import random
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from PIL import Image

from .config import Settings
from .errors import VideoQAError

QUERY = "retrieval.query"
PASSAGE = "retrieval.passage"


class Embedder(Protocol):
    model: str
    dim: int
    tokens_used: int

    async def embed_texts(
        self, texts: Sequence[str], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray: ...

    async def embed_images(
        self, images: Sequence[Path | bytes], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray: ...

    async def aclose(self) -> None: ...


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def encode_image(source: Path | bytes, max_px: int) -> str:
    """Downscale to `max_px` on the long side and return base64 JPEG."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(handle) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TokenBudget:
    """Sliding one-minute window over the provider's tokens-per-minute limit.

    A request reserves its estimated cost before it is sent and settles to the
    real cost from the response's usage report. Per-item costs are learned from
    those reports, so the estimate tracks whatever the provider actually charges
    for an image at the resolution being sent.
    """

    WINDOW = 60.0

    def __init__(self, tokens_per_minute: int, image_px: int = 448) -> None:
        self.limit = tokens_per_minute
        self._spent: list[list[float]] = []  # [timestamp, tokens]
        self._lock = asyncio.Lock()
        # jina-embeddings-v4 charges by image area: measured 840 tokens for a 16:9
        # frame at 320px, 1440 at 448px and 2990 at 640px on the long side.
        self.per_item = {"image": 0.0075 * image_px * image_px + 100.0, "text": 0.0}

    def estimate(self, kind: str, inputs: list[dict[str, str]]) -> int:
        if kind == "text":
            return max(1, int(sum(len(i.get("text", "")) for i in inputs) / 3.0) + 8 * len(inputs))
        return int(self.per_item["image"] * len(inputs))

    def learn(self, kind: str, count: int, actual: int) -> None:
        if kind == "image" and count and actual:
            self.per_item["image"] = 0.7 * self.per_item["image"] + 0.3 * (actual / count)

    async def reserve(self, tokens: int) -> list[float]:
        tokens = min(tokens, self.limit)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._spent = [e for e in self._spent if now - e[0] < self.WINDOW]
                used = sum(e[1] for e in self._spent)
                if used + tokens <= self.limit:
                    entry = [now, float(tokens)]
                    self._spent.append(entry)
                    return entry
                wait = self.WINDOW - (now - self._spent[0][0]) + 0.05
            await asyncio.sleep(min(max(wait, 0.1), 5.0))

    def settle(self, entry: list[float], actual: int | None) -> None:
        if actual is None:
            entry[1] = 0.0  # the request was not charged
        else:
            entry[1] = float(actual)

    def penalize(self, tokens: int) -> None:
        """After a 429 the provider counted more than we did; charge the rejected request anyway."""
        self._spent.append([time.monotonic(), float(tokens)])


class JinaEmbedder:
    """Jina's embeddings API. `jina-embeddings-v4` accepts text and images in one call."""

    def __init__(self, settings: Settings) -> None:
        if not settings.embed_api_key:
            raise VideoQAError(
                "no_embed_key", "set JINA_API_KEY (or DVQA_EMBED_API_KEY) to embed with jina-embeddings-v4"
            )
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.tokens_used = 0
        self._settings = settings
        self._gate = asyncio.Semaphore(settings.embed_concurrency)
        self._budget = TokenBudget(settings.embed_tokens_per_minute, settings.embed_image_px)
        self._client = httpx.AsyncClient(
            base_url=settings.embed_base_url,
            headers={"Authorization": f"Bearer {settings.embed_api_key}"},
            timeout=httpx.Timeout(180, connect=20),
            limits=httpx.Limits(max_connections=settings.embed_concurrency * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, inputs: list[dict[str, str]], task: str, kind: str) -> np.ndarray:
        body = {"model": self.model, "task": task, "dimensions": self.dim, "input": inputs}
        last = "no attempt made"
        for attempt in range(10):
            entry = await self._budget.reserve(self._budget.estimate(kind, inputs))
            charged: int | None = None
            try:
                async with self._gate:
                    response = await self._client.post("/embeddings", json=body)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    payload = response.json()
                    charged = int((payload.get("usage") or {}).get("total_tokens") or 0)
                    self._budget.settle(entry, charged)
                    self._budget.learn(kind, len(inputs), charged)
                    self.tokens_used += charged
                    rows = sorted(payload["data"], key=lambda row: row["index"])
                    matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
                    if matrix.shape != (len(inputs), self.dim):
                        raise VideoQAError(
                            "embed_failed", f"expected {(len(inputs), self.dim)} vectors, got {matrix.shape}"
                        )
                    return _unit(matrix)
                last = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    self._budget.settle(entry, None)
                    raise VideoQAError("embed_failed", f"embedding API rejected the request ({last})")
                if response.status_code == 429:
                    self._budget.penalize(int(entry[1]))
                    retry_after = response.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2.0 * (attempt + 1)
                    await asyncio.sleep(min(delay, 60) + random.random())
                    continue
            self._budget.settle(entry, charged)
            await asyncio.sleep(min(40, 1.5 * 2**attempt) + random.random())
        raise VideoQAError("embed_failed", f"embedding API kept failing ({last})")

    async def _batched(
        self,
        items: Sequence[Any],
        size: int,
        build: Callable[[Sequence[Any]], Any],
        task: str,
        kind: str,
        on_progress: Callable[[int, int], None] | None,
    ) -> np.ndarray:
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [items[i : i + size] for i in range(0, len(items), size)]
        finished = 0

        async def one(batch: Sequence[Any]) -> np.ndarray:
            nonlocal finished
            inputs = await build(batch)
            vectors = await self._post(inputs, task, kind)
            finished += len(batch)
            if on_progress:
                on_progress(finished, len(items))
            return vectors

        return np.vstack(await asyncio.gather(*(one(batch) for batch in batches)))

    async def embed_texts(
        self, texts: Sequence[str], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        async def build(batch: Sequence[str]) -> list[dict[str, str]]:
            return [{"text": text[:8000] or " "} for text in batch]

        return await self._batched(texts, self._settings.embed_text_batch, build, task, "text", on_progress)

    async def embed_images(
        self, images: Sequence[Path | bytes], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        px = self._settings.embed_image_px

        async def build(batch: Sequence[Path | bytes]) -> list[dict[str, str]]:
            encoded = await asyncio.gather(*(asyncio.to_thread(encode_image, item, px) for item in batch))
            return [{"image": data} for data in encoded]

        return await self._batched(images, self._settings.embed_image_batch, build, task, "image", on_progress)


class HashEmbedder:
    """Deterministic offline vectors for tests and dry runs. Lexical only, no real semantics.

    Text is a bag of hashed words. An image is hashed from its coarse colour
    layout, so identical pictures match and nothing else is promised.
    """

    def __init__(self, settings: Settings) -> None:
        self.model = "hash-test-embedder"
        self.dim = settings.embed_dim
        self.tokens_used = 0

    async def aclose(self) -> None:
        return None

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if value >> 63 else -1.0

    def _text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index, sign = self._bucket(token)
            vector[index] += sign
        return vector

    def _image(self, source: Path | bytes) -> np.ndarray:
        handle = io.BytesIO(source) if isinstance(source, bytes) else source
        with Image.open(handle) as image:
            small = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
        vector = np.zeros(self.dim, dtype=np.float32)
        flat = small.reshape(-1)
        for position, value in enumerate(flat):
            index, sign = self._bucket(f"px{position}")
            vector[index] += sign * (value - 0.5)
        return vector

    async def embed_texts(
        self, texts: Sequence[str], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _unit(np.vstack([self._text(text) for text in texts]))

    async def embed_images(
        self, images: Sequence[Path | bytes], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _unit(np.vstack([self._image(image) for image in images]))


def make_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "jina":
        return JinaEmbedder(settings)
    if settings.embed_backend == "hash":
        return HashEmbedder(settings)
    raise VideoQAError("config", f"unknown DVQA_EMBED_BACKEND {settings.embed_backend!r}")
