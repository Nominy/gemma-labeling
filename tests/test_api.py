from __future__ import annotations

from io import BytesIO

import httpx
from PIL import Image
import pytest

from gemma_labeling.schemas import (
    ConstrainedLabelResponse,
    FinalConstraintState,
    InvalidTokenStats,
    ParsedTagResult,
    PromptSnapshot,
    TagRuleModel,
    TaxonomyResponse,
    TraceEntry,
)
from gemma_labeling.web import create_app


class FakeRuntime:
    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_id": "fake-model",
            "cuda_available": False,
            "model_loaded": True,
            "detail": None,
        }


class FakeService:
    def __init__(self) -> None:
        self.model_runtime = FakeRuntime()

    def taxonomy_response(self) -> TaxonomyResponse:
        return TaxonomyResponse(
            tag_count=2,
            categories={"subject": 1, "detail": 1},
            starter_tags=["alpha"],
            rules=[
                TagRuleModel(canonical="alpha", category="subject"),
                TagRuleModel(canonical="gamma", category="detail", prerequisites=["alpha"]),
            ],
        )

    def label_image(self, image_bytes: bytes, **_: object) -> ConstrainedLabelResponse:
        assert image_bytes
        return ConstrainedLabelResponse(
            model_id="fake-model",
            prompt=PromptSnapshot(system_prompt="s", user_prompt="u", hint=None),
            baseline=ParsedTagResult(
                raw_text="alpha, alpha, unknown",
                normalized_tags=["alpha"],
                unknown_tags=["unknown"],
                duplicate_tags=["alpha"],
            ),
            constrained=ParsedTagResult(
                raw_text="alpha, gamma",
                normalized_tags=["alpha", "gamma"],
            ),
            trace=[
                TraceEntry(tag="alpha", unlocked_tags=["gamma"], active_tags_after=["gamma"]),
                TraceEntry(tag="gamma", unlocked_tags=[], active_tags_after=[]),
            ],
            final_state=FinalConstraintState(
                used_tags=["alpha", "gamma"],
                unlocked_tags=[],
                blocked_duplicate_tags=["alpha", "gamma"],
                current_prefix="",
                at_boundary=True,
                invalid_prefix=False,
            ),
            invalid_token_stats=InvalidTokenStats(
                total_steps=4,
                average_allowed_tokens=2.0,
                min_allowed_tokens=1,
                max_allowed_tokens=3,
                eos_allowed_steps=2,
                masked_fraction=0.8,
                per_step_allowed=[2, 1, 3, 2],
            ),
        )


@pytest.mark.anyio
async def test_taxonomy_endpoint_returns_expected_shape() -> None:
    transport = httpx.ASGITransport(app=create_app(FakeService()))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/taxonomy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tag_count"] == 2
    assert payload["starter_tags"] == ["alpha"]


@pytest.mark.anyio
async def test_label_endpoint_returns_structured_response() -> None:
    transport = httpx.ASGITransport(app=create_app(FakeService()))
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 200, 160)).save(buffer, format="PNG")
    buffer.seek(0)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/label",
            data={
                "hint": "orange portrait",
                "system_prompt": "system",
                "user_prompt": "user",
                "max_tags": "8",
                "max_new_tokens": "32",
                "temperature": "0",
            },
            files={"image": ("sample.png", buffer.getvalue(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "fake-model"
    assert payload["constrained"]["normalized_tags"] == ["alpha", "gamma"]
    assert payload["trace"][0]["unlocked_tags"] == ["gamma"]
