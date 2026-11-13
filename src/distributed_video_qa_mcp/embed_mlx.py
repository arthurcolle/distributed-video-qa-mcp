"""jina-embeddings-v4 running locally on Apple Silicon through MLX.

Uses Jina's official 8-bit MLX port (`jinaai/jina-embeddings-v4-mlx-8bit`),
pinned to a revision, with the tokenizer and image preprocessing of the original
model. No PyTorch, no network after the first download, no per-token cost.

Measured against the hosted API at 1,024 dimensions, the vectors agree at
cosine 0.999 for text and 0.988-0.995 for frames, so indexes built with either
backend are interchangeable and share the model name `jina-embeddings-v4`.

Three behaviours of the port are unsafe for a long-running server and are
avoided here:

1. Its fast `encode_text`/`encode_image` paths cache results keyed on the
   Python `id()` of the input arrays. Ids are reused once an array is freed, so
   a later call can silently return an earlier input's embedding.
2. The fast image path frees the vision encoder's weights after its first call,
   and frees the text model once both caches are filled.
3. Its compiled language-model forward ignores the padding mask and returns
   wrong vectors for some padded batches.

So the encoders run on the ordinary (uncompiled, mask-aware) path, the result
caches are cleared before every call, and the fast paths are never set up.
Verified: batched and single results agree to 1.0000 for text and images.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import platform
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import Settings
from .errors import VideoQAError

MLX_REPO = "jinaai/jina-embeddings-v4-mlx-8bit"
MLX_REVISION = "fd57282bd7a450c5db866ff23e4eb0980ba2bb2a"
BASE_REPO = "jinaai/jina-embeddings-v4"
BASE_REVISION = "853c867b65b749f3c3c72a06868140d842e04f06"
BASE_FILES = ["tokenizer.json", "preprocessor_config.json", "config.json"]

PAD_ID = 151643  # <|endoftext|>
IMAGE_PROMPT = "<|im_start|>user\n<|vision_start|>{pads}<|vision_end|>Describe the image.<|im_end|>\n"
PREFIX = {"retrieval.query": "Query: ", "retrieval.passage": "Passage: "}
TEXT_BATCH = 16
MAX_TEXT_TOKENS = 8192

_CACHES = (
    "_text_result_cache", "_text_result_cache_key", "_img_result_cache", "_img_result_cache_key_ids",
    "_img_result_cache_key_pv", "_vision_cache", "_vision_cache_key", "_img_cos_sin_cache", "_img_cos_sin_cache_key",
)  # fmt: skip


def available() -> bool:
    """True on Apple Silicon with the `mlx` extra installed."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return all(importlib.util.find_spec(m) for m in ("mlx", "tokenizers", "transformers", "huggingface_hub"))


def fetch(*, local_only: bool = False) -> tuple[Path, Path]:
    """Download (or locate) the pinned weights and preprocessing files. ~3.9 GB on first run."""
    from huggingface_hub import snapshot_download

    try:
        weights = snapshot_download(MLX_REPO, revision=MLX_REVISION, local_files_only=local_only)
        base = snapshot_download(
            BASE_REPO, revision=BASE_REVISION, allow_patterns=BASE_FILES, local_files_only=local_only
        )
    except Exception as exc:  # noqa: BLE001 - surface any hub failure as one actionable error
        hint = "run `distributed-video-qa-mcp fetch-models` while online" if local_only else str(exc)[:300]
        raise VideoQAError("config", f"jina-embeddings-v4 MLX weights are not available: {hint}") from exc
    return Path(weights), Path(base)


def cached() -> bool:
    try:
        fetch(local_only=True)
    except VideoQAError:
        return False
    return True


def _import_port(weights: Path) -> Any:
    """Import the port's model.py and load_model.py from the pinned snapshot under private module names."""
    loaded = sys.modules.get("_dvqa_jina_v4_load")
    if loaded is not None:
        return loaded
    model_spec = importlib.util.spec_from_file_location("_dvqa_jina_v4_model", weights / "model.py")
    load_spec = importlib.util.spec_from_file_location("_dvqa_jina_v4_load", weights / "load_model.py")
    assert model_spec and model_spec.loader and load_spec and load_spec.loader
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)
    # load_model.py does `from model import ...`; satisfy that one import, then remove the generic name.
    previous = sys.modules.get("model")
    sys.modules["model"] = model_module
    try:
        load_module = importlib.util.module_from_spec(load_spec)
        load_spec.loader.exec_module(load_module)
    finally:
        if previous is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous
    sys.modules["_dvqa_jina_v4_model"] = model_module
    sys.modules["_dvqa_jina_v4_load"] = load_module
    return load_module


def _load_model(weights: Path) -> Any:
    """Build the model in the port's preprocessed (fused, 8-bit) layout and load the published shards."""
    import mlx.core as mx
    import mlx.nn as nn

    port = _import_port(weights)
    model = port.JinaEmbeddingsV4(port.load_config(str(weights)))
    port._fuse_qkv_projections(model)
    port._pad_vision_mlp_weights(model)
    port._fuse_gate_up_projections(model)

    def quantizable(_: str, module: Any) -> bool:
        if not isinstance(module, (nn.Linear, port.LoRALinear)):
            return False
        out_dim, in_dim = module.weight.shape
        return in_dim % 128 == 0 and out_dim % 128 == 0

    nn.quantize(model, group_size=128, bits=8, class_predicate=quantizable)
    model.language_model.embed_tokens = nn.QuantizedEmbedding.from_embedding(
        model.language_model.embed_tokens, group_size=128, bits=8
    )
    index = json.loads((weights / "model.safetensors.index.json").read_text())
    tensors: dict[str, Any] = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(mx.load(str(weights / shard)))
    model.load_weights(list(tensors.items()), strict=True)
    mx.eval(model.parameters())
    # Deliberately no setup_compile / setup_encode_*_compile: see the module docstring.
    model._compiled_encode_text = None
    model._compiled_encode_image_text = None
    model.language_model._compiled_forward = None
    model.visual._compiled_forward = None
    return model


class MlxJinaEmbedder:
    """Same interface as the API embedder; all MLX work runs on one dedicated thread."""

    backend = "mlx"

    def __init__(self, settings: Settings) -> None:
        if not available():
            raise VideoQAError("config", "the MLX backend needs Apple Silicon and the 'mlx' extra")
        if settings.embed_dim > 2048:
            raise VideoQAError("config", "jina-embeddings-v4 produces at most 2048 dimensions")
        self.model = "jina-embeddings-v4"
        self.dim = settings.embed_dim
        self.tokens_used = 0
        self._image_px = settings.embed_image_px
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-embed")
        self._ready = threading.Event()
        self._state: dict[str, Any] = {}

    # ------------------------------------------------------------ loading

    def _ensure(self) -> dict[str, Any]:
        if self._ready.is_set():
            return self._state
        from tokenizers import Tokenizer
        from transformers import Qwen2VLImageProcessorPil as Qwen2VLImageProcessor

        weights, base = fetch()
        self._state = {
            "model": _load_model(weights),
            "tokenizer": Tokenizer.from_file(str(base / "tokenizer.json")),
            "images": Qwen2VLImageProcessor.from_pretrained(str(base)),
        }
        self._ready.set()
        return self._state

    async def _run(self, fn: Callable[..., np.ndarray], *args: Any) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(self._thread, fn, *args)

    async def aclose(self) -> None:
        self._thread.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ encoding (runs on the MLX thread)

    @staticmethod
    def _batch(sequences: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(s) for s in sequences)
        ids = np.full((len(sequences), width), PAD_ID, dtype=np.int32)
        mask = np.zeros((len(sequences), width), dtype=np.int32)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        return ids, mask

    def _finish(self, vectors: Any) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)[:, : self.dim]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def _encode_texts(self, texts: Sequence[str], task: str) -> np.ndarray:
        import mlx.core as mx

        state = self._ensure()
        model, tokenizer = state["model"], state["tokenizer"]
        prefix = PREFIX.get(task, "")
        encoded = [tokenizer.encode(prefix + (text or " ")).ids[:MAX_TEXT_TOKENS] for text in texts]
        self.tokens_used += sum(len(e) for e in encoded)
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))  # less padding per batch
        out = np.zeros((len(encoded), self.dim), dtype=np.float32)
        for start in range(0, len(order), TEXT_BATCH):
            rows = order[start : start + TEXT_BATCH]
            ids, mask = self._batch([encoded[i] for i in rows])
            for name in _CACHES:
                setattr(model, name, None)
            vectors = model.encode_text(input_ids=mx.array(ids), attention_mask=mx.array(mask), task="retrieval")
            mx.eval(vectors)
            out[rows] = self._finish(vectors)
        return out

    def _encode_image(self, source: Path | bytes) -> np.ndarray:
        import io

        import mlx.core as mx

        state = self._ensure()
        model, tokenizer, processor = state["model"], state["tokenizer"], state["images"]
        handle = io.BytesIO(source) if isinstance(source, bytes) else source
        with Image.open(handle) as image:
            image = image.convert("RGB")
            image.thumbnail((self._image_px, self._image_px), Image.Resampling.LANCZOS)
            pixels = processor(images=[image], return_tensors="np")
        grid = pixels["image_grid_thw"][0]
        pads = "<|image_pad|>" * (int(np.prod(grid)) // 4)  # 2x2 spatial merge
        tokens = tokenizer.encode(IMAGE_PROMPT.format(pads=pads)).ids
        self.tokens_used += len(tokens)
        ids, mask = self._batch([tokens])
        for name in _CACHES:
            setattr(model, name, None)
        vectors = model.encode_image(
            input_ids=mx.array(ids),
            pixel_values=mx.array(pixels["pixel_values"]),
            image_grid_thw=[tuple(int(x) for x in grid)],
            attention_mask=mx.array(mask),
            task="retrieval",
        )
        mx.eval(vectors)
        return self._finish(vectors)[0]

    # ------------------------------------------------------------ public interface

    async def embed_texts(
        self, texts: Sequence[str], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        step = TEXT_BATCH * 4
        for start in range(0, len(texts), step):
            out.append(await self._run(self._encode_texts, list(texts[start : start + step]), task))
            if on_progress:
                on_progress(min(start + step, len(texts)), len(texts))
        return np.vstack(out)

    async def embed_images(
        self, images: Sequence[Path | bytes], task: str, on_progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for index, image in enumerate(images):
            out[index] = await self._run(self._encode_image, image)
            if on_progress:
                on_progress(index + 1, len(images))
        return out
