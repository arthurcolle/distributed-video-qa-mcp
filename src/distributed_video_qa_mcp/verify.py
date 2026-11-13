"""Optional evidence verification with Jev 1.14 (TypeSafe System One).

Retrieval similarity says what is *near* a question, not whether it *answers* it.
On a test set from one video, the best transcript similarity was 0.51-0.76 for
questions the video answers and 0.42-0.55 for questions it does not: the ranges
overlap, so no threshold separates them. Jev's probability that a moment answers
the question was at least 0.92 for the answerable questions whose answer
retrieval had found, and at most 0.04 for every unanswerable one.

For each candidate moment the verifier asks two typed yes/no (Noul) questions:
does the transcript window answer the question, and does each frame show what
was asked. Moments are reranked by their strongest support. When nothing is
supported, a frames-only search runs as a rescue, because some answers are only
on screen (or mis-transcribed, as "openmedio.com" for "open-meteo.com").

Off by default: verification sends transcript excerpts and frames to TypeSafe.
Enable per call with `verify=true`, or by default with `DVQA_VERIFY=jev`.
Probabilities are returned as-is; the 0.5 threshold for `supported` is this
server's policy, not part of the model's answer.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from .config import Settings
from .errors import VideoQAError

SUPPORT_THRESHOLD = 0.5


def available(settings: Settings) -> bool:
    return bool(settings.typesafe_api_key) and importlib.util.find_spec("typesafe_sdk") is not None


def describe(settings: Settings) -> str:
    if importlib.util.find_spec("typesafe_sdk") is None:
        return "unavailable (install the 'jev' extra)"
    if not settings.typesafe_api_key:
        return "unavailable (set TYPESAFE_API_KEY)"
    default = "on by default" if settings.verify_default else "per call (verify=true)"
    return f"{settings.verify_model} via TypeSafe, {default}"


@dataclass
class Judgement:
    answers: float | None = None  # p(the transcript window answers the question)
    frames: dict[str, float] = field(default_factory=dict)  # frame path -> p(frame shows what was asked)

    @property
    def support(self) -> float:
        return max([self.answers or 0.0, *self.frames.values()])


class JevVerifier:
    def __init__(self, settings: Settings) -> None:
        if not available(settings):
            raise VideoQAError("verify_unavailable", describe(settings))
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        self.model = settings.verify_model
        self._client = AsyncTypeSafeClient(
            api_key=settings.typesafe_api_key, model=self.model, retry=RetryPolicy(max_retries=2), timeout=60
        )
        self._gate = asyncio.Semaphore(8)
        self.input_tokens = 0
        self.calls = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _ask(self, state: Any, instructions: str, *, image: bool = False) -> float:
        from typesafe_sdk import Noul

        headers = {"TypeSafe-Danger-Zone-Images": "true"} if image else None
        async with self._gate:
            response = await self._client.system_one(
                state=state, questions={"yes": Noul(instructions=instructions)}, extra_headers=headers
            )
        if response.model != self.model:
            raise VideoQAError("verify_failed", f"asked for {self.model}, the provider answered with {response.model}")
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        return float(response.answers["yes"].noul)

    async def transcript_answers(self, question: str, lines: list[dict[str, Any]]) -> float | None:
        if not lines:
            return None
        state = {
            "question": question,
            "evidence": [{"at": line["at"], "said": line["text"]} for line in lines],
            "instruction": "Judge only from the evidence lines.",
        }
        return await self._ask(state, "Does the evidence contain a direct answer to the question?")

    async def frame_shows(self, question: str, path: str | Path) -> float:
        from typesafe_sdk import Image

        def load() -> Any:
            with PILImage.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((768, 768))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
            return PILImage.open(io.BytesIO(buffer.getvalue()))

        picture = await asyncio.to_thread(load)
        instructions = (
            f"Question about a video: {question!r}. "
            "Does this frame show something that answers it or that it asks to see?"
        )
        return await self._ask(Image.from_pil(picture, format="jpeg"), instructions, image=True)

    async def judge(self, question: str, transcript: list[dict[str, Any]], frame_paths: list[str]) -> Judgement:
        tasks = [self.transcript_answers(question, transcript)]
        tasks += [self.frame_shows(question, path) for path in frame_paths]
        results = await asyncio.gather(*tasks)
        return Judgement(answers=results[0], frames=dict(zip(frame_paths, results[1:], strict=True)))
