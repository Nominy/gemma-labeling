from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from gemma_labeling.model import GemmaModelRuntime
from gemma_labeling.taxonomy import Taxonomy


@pytest.fixture()
def taxonomy() -> Taxonomy:
    return Taxonomy.from_records(
        [
            {"canonical": "alpha", "category": "subject"},
        ]
    )


def test_runtime_hard_fails_without_cuda(monkeypatch: pytest.MonkeyPatch, taxonomy: Taxonomy) -> None:
    monkeypatch.setattr("gemma_labeling.model.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("gemma_labeling.model.torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr("gemma_labeling.model.torch.version.cuda", None)

    with pytest.raises(RuntimeError, match="CUDA is required for this app"):
        GemmaModelRuntime(taxonomy, require_cuda=True)


def test_runtime_can_be_forced_to_cpu_for_debug(monkeypatch: pytest.MonkeyPatch, taxonomy: Taxonomy) -> None:
    monkeypatch.setattr("gemma_labeling.model.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("gemma_labeling.model.torch.cuda.device_count", lambda: 0)
    monkeypatch.setattr("gemma_labeling.model.torch.version.cuda", None)

    runtime = GemmaModelRuntime(taxonomy, require_cuda=False)
    assert runtime._device.type == "cpu"


def test_runtime_selects_gguf_backend_for_local_model(tmp_path: Path, taxonomy: Taxonomy) -> None:
    model_path = tmp_path / "gemma-4-E4B-it-Q4_K_M.gguf"
    mmproj_path = tmp_path / "mmproj-F16.gguf"
    model_path.write_text("stub", encoding="utf-8")
    mmproj_path.write_text("stub", encoding="utf-8")

    runtime = GemmaModelRuntime(
        taxonomy,
        model_id=str(model_path),
        backend="gguf",
        gguf_mmproj_path=str(mmproj_path),
        require_cuda=False,
        llama_server_auto_start=False,
    )

    assert runtime.backend_name == "gguf"
    assert runtime.model_id == str(model_path)


def test_gguf_runtime_requires_mmproj_for_multimodal_requests(
    tmp_path: Path,
    taxonomy: Taxonomy,
) -> None:
    model_path = tmp_path / "gemma-4-E4B-it-Q4_K_M.gguf"
    model_path.write_text("stub", encoding="utf-8")

    runtime = GemmaModelRuntime(
        taxonomy,
        model_id=str(model_path),
        backend="gguf",
        gguf_mmproj_path=None,
        require_cuda=False,
        llama_server_auto_start=False,
    )

    with pytest.raises(RuntimeError, match="GEMMA_LABELING_GGUF_MMPROJ_PATH"):
        runtime.generate(
            Image.new("RGB", (4, 4), color=(255, 255, 255)),
            system_prompt="system",
            user_prompt="user",
            max_tags=4,
            max_new_tokens=16,
            temperature=0.0,
            constrained=False,
        )
