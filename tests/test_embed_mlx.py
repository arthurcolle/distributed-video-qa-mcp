"""Local MLX embedder checks. Run on Apple Silicon with the weights cached; skipped elsewhere.

These pin down the failure modes found in the upstream MLX port: results reused
across calls through id()-keyed caches, vision weights freed after one image, and
padded batches that disagree with single inputs.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from distributed_video_qa_mcp import embed_mlx
from distributed_video_qa_mcp.config import Settings
from distributed_video_qa_mcp.embed import PASSAGE, QUERY

pytestmark = pytest.mark.skipif(
    not (embed_mlx.available() and embed_mlx.cached()),
    reason="needs Apple Silicon, the mlx extra and cached weights (distributed-video-qa-mcp fetch-models)",
)


def picture(colour: str, shape: str) -> bytes:
    image = Image.new("RGB", (448, 252), "white")
    draw = ImageDraw.Draw(image)
    (draw.ellipse if shape == "circle" else draw.rectangle)([150, 50, 300, 200], fill=colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    return embed_mlx.MlxJinaEmbedder(Settings(state_dir=tmp_path_factory.mktemp("s"), embed_backend="mlx"))


async def test_distinct_images_get_distinct_vectors(embedder):
    pictures = [picture(c, s) for c in ("red", "blue", "green") for s in ("circle", "square")]
    first = await embedder.embed_images(pictures, PASSAGE)
    again = await embedder.embed_images(list(reversed(pictures)), PASSAGE)
    similarity = first @ first.T
    assert first.shape == (6, embedder.dim)
    assert similarity[np.triu_indices(6, 1)].max() < 0.99, "two different images produced the same vector"
    assert np.allclose(first, again[::-1], atol=1e-3), "the same image embedded differently on a second call"


async def test_text_batches_match_single_inputs(embedder):
    texts = ["a blue whale eats krill", "Seoul's temperature crept up all day and hit exactly 13 degrees", "hi"]
    batched = await embedder.embed_texts(texts, PASSAGE)
    singles = np.vstack([await embedder.embed_texts([t], PASSAGE) for t in texts])
    assert np.allclose(batched, singles, atol=2e-3)


async def test_retrieval_ranks_the_right_passage_and_picture(embedder):
    passages = ["Rovers measure wind on Mars.", "Blue whales are the largest animals.", "Leaves turn light into sugar."]
    doc = await embedder.embed_texts(passages, PASSAGE)
    query = (await embedder.embed_texts(["what is the biggest animal"], QUERY))[0]
    assert int(np.argmax(doc @ query)) == 1
    pictures = await embedder.embed_images([picture("red", "circle"), picture("blue", "square")], PASSAGE)
    ask = (await embedder.embed_texts(["a blue square"], QUERY))[0]
    assert int(np.argmax(pictures @ ask)) == 1
