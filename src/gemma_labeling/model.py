from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from PIL import Image
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from gemma_labeling.config import DEFAULT_MODEL_ID, DEFAULT_REQUIRE_CUDA
from gemma_labeling.constraints import ConstrainedTagLogitsProcessor, ConstraintRuntime
from gemma_labeling.taxonomy import Taxonomy


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    stats: dict[str, float | int | list[int]]
    snapshot: dict[str, object]


class GemmaModelRuntime:
    def __init__(
        self,
        taxonomy: Taxonomy,
        model_id: str = DEFAULT_MODEL_ID,
        require_cuda: bool = DEFAULT_REQUIRE_CUDA,
    ) -> None:
        self.taxonomy = taxonomy
        self.model_id = model_id
        self.require_cuda = require_cuda
        self._processor = None
        self._model = None
        self._constraint_runtime = None
        self._lock = Lock()
        self._load_error: str | None = None
        self._device = self._detect_device()

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def health(self) -> dict[str, object]:
        status = "ok" if self.load_error is None else "degraded"
        return {
            "status": status,
            "model_id": self.model_id,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.loaded,
            "detail": self.load_error or f"device={self._device.type}",
        }

    def generate(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        constrained: bool,
    ) -> GenerationResult:
        with self._lock:
            self._ensure_loaded()
            assert self._processor is not None
            assert self._model is not None
            assert self._constraint_runtime is not None

            inputs = self._build_inputs(image, system_prompt=system_prompt, user_prompt=user_prompt)
            prompt_length = int(inputs["input_ids"].shape[-1])
            for name, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[name] = value.to(self._device)

            processor = (
                ConstrainedTagLogitsProcessor(self._constraint_runtime, prompt_length)
                if constrained
                else None
            )

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": self._processor.tokenizer.pad_token_id
                or self._processor.tokenizer.eos_token_id,
            }
            if processor is not None:
                generation_kwargs["logits_processor"] = [processor]

            outputs = self._model.generate(
                **inputs,
                **{key: value for key, value in generation_kwargs.items() if value is not None},
            )

            new_tokens = outputs[0, prompt_length:]
            text = self._processor.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if processor is None:
                return GenerationResult(text=text, stats={}, snapshot={})

            snapshot = {
                "used_tags": sorted(processor.last_snapshot.state.used_tags),
                "unlocked_tags": sorted(processor.last_snapshot.state.unlocked_tags),
                "blocked_duplicate_tags": sorted(processor.last_snapshot.state.used_tags),
                "current_prefix": processor.last_snapshot.current_prefix,
                "at_boundary": processor.last_snapshot.at_boundary,
                "invalid_prefix": processor.last_snapshot.invalid_prefix,
            }
            return GenerationResult(
                text=text,
                stats=self._constraint_runtime.summarize_steps(processor.step_stats),
                snapshot=snapshot,
            )

    def _ensure_loaded(self) -> None:
        if self.loaded or self._load_error is not None:
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            return

        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            model_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                dtype=model_dtype,
            )
            model.to(self._device)
            model.eval()

            self._processor = processor
            self._model = model
            self._constraint_runtime = ConstraintRuntime(processor.tokenizer, self.taxonomy)
        except Exception as exc:  # pragma: no cover - depends on remote auth/model access
            self._load_error = (
                "Failed to load the Gemma model. Confirm that you have access to "
                f"`{self.model_id}` and that the environment has a Gemma 4-compatible "
                "Transformers build installed. Original error: "
                f"{exc}"
            )
            raise RuntimeError(self._load_error) from exc

    def _detect_device(self) -> torch.device:
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count() if cuda_available else 0

        if self.require_cuda and (not cuda_available or cuda_version is None or device_count < 1):
            raise RuntimeError(
                "CUDA is required for this app, but the active Python environment is not CUDA-capable. "
                f"torch={torch.__version__}, torch.version.cuda={cuda_version}, "
                f"cuda.is_available={cuda_available}, device_count={device_count}. "
                "Install CUDA-enabled PyTorch wheels in the project venv and restart the server."
            )

        return torch.device("cuda" if cuda_available else "cpu")

    def _build_inputs(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        assert self._processor is not None
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
