from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gemma_labeling.config import UI_ROOT
from gemma_labeling.schemas import ConstrainedLabelResponse, HealthResponse, TaxonomyResponse
from gemma_labeling.service import LabelingService


def create_app(service: LabelingService) -> FastAPI:
    app = FastAPI(title="Gemma Labeling PoC", version="0.1.0")
    app.mount("/static", StaticFiles(directory=UI_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(UI_ROOT / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(**service.model_runtime.health())

    @app.get("/taxonomy", response_model=TaxonomyResponse)
    def taxonomy() -> TaxonomyResponse:
        return service.taxonomy_response()

    @app.post("/label", response_model=ConstrainedLabelResponse)
    async def label(
        image: UploadFile = File(...),
        hint: str | None = Form(default=None),
        system_prompt: str | None = Form(default=None),
        user_prompt: str | None = Form(default=None),
        max_tags: int = Form(default=12),
        max_new_tokens: int = Form(default=48),
        temperature: float = Form(default=0.0),
    ) -> ConstrainedLabelResponse:
        image_bytes = await image.read()
        return await run_in_threadpool(
            service.label_image,
            image_bytes,
            hint=hint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tags=max_tags,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return app
