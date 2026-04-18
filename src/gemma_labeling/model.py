from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
import shutil
import subprocess
import time
from typing import Any, Protocol

import httpx
from PIL import Image
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from gemma_labeling.config import (
    DEFAULT_BACKEND,
    DEFAULT_GGUF_MMPROJ_PATH,
    DEFAULT_LLAMA_SERVER_ALIAS,
    DEFAULT_LLAMA_SERVER_AUTO_START,
    DEFAULT_LLAMA_SERVER_BIN,
    DEFAULT_LLAMA_SERVER_CTX_SIZE,
    DEFAULT_LLAMA_SERVER_N_GPU_LAYERS,
    DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT,
    DEFAULT_LLAMA_SERVER_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_REQUIRE_CUDA,
)
from gemma_labeling.constraints import ConstrainedTagLogitsProcessor, ConstraintRuntime
from gemma_labeling.taxonomy import TagState, Taxonomy


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    stats: dict[str, float | int | list[int]]
    snapshot: dict[str, object]


class RuntimeBackend(Protocol):
    model_id: str
    backend_name: str

    @property
    def loaded(self) -> bool: ...

    @property
    def load_error(self) -> str | None: ...

    def health(self) -> dict[str, object]: ...

    def generate(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tags: int,
        max_new_tokens: int,
        temperature: float,
        constrained: bool,
    ) -> GenerationResult: ...


class GemmaModelRuntime:
    def __init__(
        self,
        taxonomy: Taxonomy,
        model_id: str = DEFAULT_MODEL_ID,
        require_cuda: bool = DEFAULT_REQUIRE_CUDA,
        backend: str | None = DEFAULT_BACKEND,
        gguf_mmproj_path: str | None = DEFAULT_GGUF_MMPROJ_PATH,
        llama_server_url: str = DEFAULT_LLAMA_SERVER_URL,
        llama_server_bin: str = DEFAULT_LLAMA_SERVER_BIN,
        llama_server_auto_start: bool = DEFAULT_LLAMA_SERVER_AUTO_START,
        llama_server_n_gpu_layers: int = DEFAULT_LLAMA_SERVER_N_GPU_LAYERS,
        llama_server_ctx_size: int = DEFAULT_LLAMA_SERVER_CTX_SIZE,
        llama_server_startup_timeout: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT,
        llama_server_alias: str = DEFAULT_LLAMA_SERVER_ALIAS,
    ) -> None:
        resolved_backend = _resolve_backend_name(backend, model_id)
        if resolved_backend == "gguf":
            self._backend: RuntimeBackend = GGUFServerBackend(
                taxonomy=taxonomy,
                model_id=model_id,
                require_cuda=require_cuda,
                gguf_mmproj_path=gguf_mmproj_path,
                server_url=llama_server_url,
                server_bin=llama_server_bin,
                auto_start=llama_server_auto_start,
                n_gpu_layers=llama_server_n_gpu_layers,
                ctx_size=llama_server_ctx_size,
                startup_timeout=llama_server_startup_timeout,
                server_alias=llama_server_alias,
            )
        else:
            self._backend = TransformersGemmaBackend(
                taxonomy=taxonomy,
                model_id=model_id,
                require_cuda=require_cuda,
            )

        self.model_id = self._backend.model_id
        self.backend_name = self._backend.backend_name
        self._device = getattr(self._backend, "_device", None)

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

    @property
    def load_error(self) -> str | None:
        return self._backend.load_error

    def health(self) -> dict[str, object]:
        return self._backend.health()

    def generate(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tags: int,
        max_new_tokens: int,
        temperature: float,
        constrained: bool,
    ) -> GenerationResult:
        return self._backend.generate(
            image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tags=max_tags,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            constrained=constrained,
        )


class TransformersGemmaBackend:
    backend_name = "transformers"

    def __init__(
        self,
        taxonomy: Taxonomy,
        model_id: str,
        require_cuda: bool,
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
            "detail": self.load_error or f"backend=transformers device={self._device.type}",
        }

    def generate(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tags: int,
        max_new_tokens: int,
        temperature: float,
        constrained: bool,
    ) -> GenerationResult:
        del max_tags

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


class GGUFServerBackend:
    backend_name = "gguf"

    def __init__(
        self,
        taxonomy: Taxonomy,
        model_id: str,
        require_cuda: bool,
        gguf_mmproj_path: str | None,
        server_url: str,
        server_bin: str,
        auto_start: bool,
        n_gpu_layers: int,
        ctx_size: int,
        startup_timeout: float,
        server_alias: str,
    ) -> None:
        self.taxonomy = taxonomy
        self.model_path = Path(model_id).expanduser()
        self.model_id = str(self.model_path)
        self.require_cuda = require_cuda
        self.mmproj_path = Path(gguf_mmproj_path).expanduser() if gguf_mmproj_path else None
        self.server_url = server_url.rstrip("/")
        self.server_bin = server_bin
        self.auto_start = auto_start
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.startup_timeout = startup_timeout
        self.server_alias = server_alias
        self._lock = Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._server_process: subprocess.Popen[str] | None = None
        self._server_logs: deque[str] = deque(maxlen=200)
        self._saw_cuda_log = False
        self._client = httpx.Client(base_url=self.server_url, timeout=httpx.Timeout(120.0))

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def health(self) -> dict[str, object]:
        ready = self._server_ready(timeout=2.0)
        status = "ok" if self.load_error is None else "degraded"
        return {
            "status": status,
            "model_id": self.model_id,
            "cuda_available": self._saw_cuda_log or _host_cuda_available(),
            "model_loaded": self.loaded and ready,
            "detail": self.load_error or f"backend=gguf server={self.server_url} ready={ready}",
        }

    def generate(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tags: int,
        max_new_tokens: int,
        temperature: float,
        constrained: bool,
    ) -> GenerationResult:
        with self._lock:
            self._ensure_loaded()
            if constrained:
                return self._generate_constrained(
                    image,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tags=max_tags,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            return self._generate_baseline(
                image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def _ensure_loaded(self) -> None:
        if self._loaded or self._load_error is not None:
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            return

        try:
            self._validate_paths()
            if not self._server_ready(timeout=2.0):
                if not self.auto_start:
                    raise RuntimeError(
                        "No llama-server instance is reachable. Either start llama-server manually "
                        f"at `{self.server_url}` or set GEMMA_LABELING_LLAMA_SERVER_AUTO_START=1."
                    )
                self._launch_server()

            if not self._server_ready(timeout=self.startup_timeout):
                raise RuntimeError(
                    "Timed out waiting for llama-server to become ready at "
                    f"`{self.server_url}`."
                )

            if self.require_cuda and self.auto_start:
                self._wait_for_cuda_log(timeout=3.0)
            if self.require_cuda and not self._saw_cuda_log and self.auto_start:
                raise RuntimeError(
                    "GEMMA_LABELING_REQUIRE_CUDA=1 is set, but the auto-started llama-server "
                    "did not report CUDA initialization. Install a CUDA-enabled llama.cpp build "
                    "or point GEMMA_LABELING_LLAMA_SERVER_BIN at one."
                )

            self._loaded = True
        except Exception as exc:
            self._load_error = (
                "Failed to initialize the GGUF backend. Confirm that the GGUF model path, "
                "mmproj path, and llama-server configuration are correct. Original error: "
                f"{exc}"
            )
            raise RuntimeError(self._load_error) from exc

    def _validate_paths(self) -> None:
        if self.model_path.suffix.lower() != ".gguf":
            raise RuntimeError(
                "GGUF mode expects GEMMA_LABELING_MODEL_ID to point to a local `.gguf` file."
            )
        if not self.model_path.is_file():
            raise RuntimeError(f"GGUF model file does not exist: `{self.model_path}`")
        if self.mmproj_path is None:
            raise RuntimeError(
                "This app is multimodal, so GEMMA_LABELING_GGUF_MMPROJ_PATH is required in GGUF mode."
            )
        if self.mmproj_path.suffix.lower() != ".gguf" or not self.mmproj_path.is_file():
            raise RuntimeError(f"GGUF mmproj file does not exist: `{self.mmproj_path}`")
        if self.require_cuda and self.n_gpu_layers < 1:
            raise RuntimeError(
                "GEMMA_LABELING_REQUIRE_CUDA=1 requires GEMMA_LABELING_LLAMA_SERVER_N_GPU_LAYERS "
                "to be greater than 0."
            )

    def _launch_server(self) -> None:
        if self._server_process is not None and self._server_process.poll() is None:
            return

        binary = shutil.which(self.server_bin) or self.server_bin
        if not Path(binary).exists() and shutil.which(binary) is None:
            raise RuntimeError(
                "Could not find `llama-server`. Install llama.cpp or set "
                "GEMMA_LABELING_LLAMA_SERVER_BIN to the full executable path."
            )

        host, port = _parse_server_binding(self.server_url)
        command = [
            binary,
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--host",
            host,
            "--port",
            str(port),
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.ctx_size),
            "--alias",
            self.server_alias,
        ]

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        Thread(target=self._consume_server_logs, daemon=True).start()

    def _consume_server_logs(self) -> None:
        assert self._server_process is not None
        assert self._server_process.stdout is not None
        for line in self._server_process.stdout:
            cleaned = line.strip()
            if not cleaned:
                continue
            self._server_logs.append(cleaned)
            lowered = cleaned.lower()
            if "cuda" in lowered and (
                "ggml_cuda_init" in lowered
                or "loaded cuda backend" in lowered
                or "cuda devices" in lowered
            ):
                self._saw_cuda_log = True

    def _wait_for_cuda_log(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._saw_cuda_log:
                return
            if self._server_process is not None and self._server_process.poll() is not None:
                return
            time.sleep(0.1)

    def _server_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = self._client.get("/v1/models", timeout=2.0)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            if self._server_process is not None and self._server_process.poll() is not None:
                return False

            time.sleep(0.25)

        return False

    def _generate_baseline(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        text = self._chat_once(
            image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return GenerationResult(text=text, stats={}, snapshot={})

    def _generate_constrained(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tags: int,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        state = self.taxonomy.initial_state()
        selected_tags: list[str] = []
        allowed_per_step: list[int] = []

        for _ in range(max_tags):
            legal_tags = self.taxonomy.legal_tags(state)
            options = [*legal_tags, "<END>"]
            allowed_per_step.append(len(options))
            choice = self._choose_next_tag(
                image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                selected_tags=selected_tags,
                options=options,
                max_tokens=min(max_new_tokens, 32),
                temperature=temperature,
            )
            if choice == "<END>":
                break

            state, _, _ = self.taxonomy.apply_tag(state, choice)
            selected_tags.append(choice)

        return GenerationResult(
            text=", ".join(selected_tags),
            stats=_summarize_choice_counts(allowed_per_step),
            snapshot=_snapshot_from_state(state),
        )

    def _choose_next_tag(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        selected_tags: list[str],
        options: list[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        state_block = ", ".join(selected_tags) if selected_tags else "(none)"
        step_system_prompt = (
            f"{system_prompt}\n"
            "For this step, return exactly one item from the active grammar. "
            "Do not output commas, prose, explanations, or multiple tags."
        )
        step_user_prompt = (
            f"{user_prompt}\n\n"
            f"Already selected tags: {state_block}\n"
            "Choose the single best next canonical tag for this image. "
            "If no more legal tags should be added, return <END>."
        )
        text = self._chat_once(
            image,
            system_prompt=step_system_prompt,
            user_prompt=step_user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=_choice_grammar(options),
        ).strip()

        if text in options:
            return text

        normalized = text.strip(" \t\r\n,.;:")
        if normalized in options:
            return normalized

        raise RuntimeError(
            "llama-server returned a value outside the constrained option set. "
            f"Received `{text}`; first options were {options[:8]}."
        )

    def _chat_once(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        grammar: str | None = None,
    ) -> str:
        image_data_uri = _image_to_data_uri(image)
        last_error: str | None = None
        for variant in ("image_url", "image_url_short", "image_inline"):
            payload = {
                "model": self.server_alias,
                "messages": _build_server_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_data_uri=image_data_uri,
                    image_variant=variant,
                ),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if grammar is not None:
                payload["grammar"] = grammar

            try:
                response = self._client.post("/v1/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Failed to call llama-server at `{self.server_url}`: {exc}"
                ) from exc

            if response.status_code == 200:
                return _extract_response_text(response.json())

            body = response.text.strip()
            last_error = body or f"HTTP {response.status_code}"
            if "Unsupported content part type" in body or "Failed to parse messages" in body:
                continue
            if grammar is not None and "grammar" in body.lower():
                raise RuntimeError(
                    "llama-server rejected the `grammar` request parameter. "
                    "Use a recent llama.cpp build with structured output support."
                )
            raise RuntimeError(
                f"llama-server returned HTTP {response.status_code}: {body}"
            )

        raise RuntimeError(
            "llama-server did not accept any of the multimodal message payload variants. "
            f"Last error: {last_error}"
        )


def _resolve_backend_name(backend: str | None, model_id: str) -> str:
    if backend:
        lowered = backend.strip().lower()
        if lowered in {"gguf", "llama.cpp", "llama_cpp"}:
            return "gguf"
        if lowered in {"transformers", "hf", "huggingface"}:
            return "transformers"
        raise ValueError(
            f"Unsupported backend `{backend}`. Use `transformers` or `gguf`."
        )
    return "gguf" if model_id.lower().endswith(".gguf") else "transformers"


def _host_cuda_available() -> bool:
    return torch.cuda.is_available()


def _parse_server_binding(server_url: str) -> tuple[str, int]:
    if "://" not in server_url:
        raise RuntimeError(
            "GEMMA_LABELING_LLAMA_SERVER_URL must include a scheme such as http://127.0.0.1:8081"
        )
    _, remainder = server_url.split("://", 1)
    if "/" in remainder:
        remainder = remainder.split("/", 1)[0]
    host, _, port_text = remainder.partition(":")
    if not host or not port_text:
        raise RuntimeError(
            "GEMMA_LABELING_LLAMA_SERVER_URL must include both host and port, "
            f"got `{server_url}`"
        )
    return host, int(port_text)


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_server_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    image_data_uri: str,
    image_variant: str,
) -> list[dict[str, object]]:
    image_part: dict[str, object]
    if image_variant == "image_url":
        image_part = {"type": "image_url", "image_url": {"url": image_data_uri}}
    elif image_variant == "image_url_short":
        image_part = {"type": "image_url", "url": image_data_uri}
    else:
        image_part = {"type": "image", "url": image_data_uri}

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _extract_response_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected llama-server response: {payload}")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"Unexpected llama-server choice payload: {payload}")

    if isinstance(choice.get("text"), str):
        return str(choice["text"]).strip()

    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected llama-server message payload: {payload}")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return " ".join(part for part in text_parts if part).strip()

    raise RuntimeError(f"Unsupported llama-server content payload: {payload}")


def _choice_grammar(options: list[str]) -> str:
    literals = " | ".join(f'"{_escape_grammar_literal(option)}"' for option in options)
    return f"root ::= {literals}\n"


def _escape_grammar_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _snapshot_from_state(state: TagState) -> dict[str, object]:
    return {
        "used_tags": sorted(state.used_tags),
        "unlocked_tags": sorted(state.unlocked_tags),
        "blocked_duplicate_tags": sorted(state.used_tags),
        "current_prefix": "",
        "at_boundary": True,
        "invalid_prefix": False,
    }


def _summarize_choice_counts(counts: list[int]) -> dict[str, float | int | list[int]]:
    if not counts:
        return {
            "total_steps": 0,
            "average_allowed_tokens": 0.0,
            "min_allowed_tokens": 0,
            "max_allowed_tokens": 0,
            "eos_allowed_steps": 0,
            "masked_fraction": 0.0,
            "per_step_allowed": [],
        }

    return {
        "total_steps": len(counts),
        "average_allowed_tokens": sum(counts) / len(counts),
        "min_allowed_tokens": min(counts),
        "max_allowed_tokens": max(counts),
        "eos_allowed_steps": len(counts),
        "masked_fraction": 0.0,
        "per_step_allowed": counts,
    }
