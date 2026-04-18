from __future__ import annotations

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
