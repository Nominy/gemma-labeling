# Gemma Labeling PoC

Local FastAPI proof of concept for Gemma 4 E2B image labeling with hard decode-time tag constraints and an e621-derived tag grammar snapshot.

## What it demonstrates

- Baseline multimodal generation from an uploaded image.
- Constrained decoding over an e621-derived booru-style taxonomy.
- No duplicate tags.
- Alias normalization and implication closure.
- Debug views for the final FSM state and per-step masking stats.

## Stack

- `uv` for project management and execution
- `FastAPI` for the web app
- `transformers` + `torch` for local Gemma inference
- `llama.cpp` `llama-server` for local GGUF inference
- custom `LogitsProcessor` + token tries for decode-time masking

## Setup

1. Accept access to `google/gemma-4-E2B-it` on Hugging Face if the checkpoint requires it.
2. Export a Hugging Face token if your environment needs one:

   ```powershell
   $env:HF_TOKEN = "your-token"
   ```

3. Sync the environment:

   ```powershell
   uv sync --group dev
   ```

4. Run the app:

   ```powershell
   uv run gemma-labeling --reload
   ```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Environment file

An example config lives at `.env.example`.

- `transformers` backend: set `GEMMA_LABELING_BACKEND=transformers` and point `GEMMA_LABELING_MODEL_ID` at a Hugging Face model ID or a local Transformers checkpoint directory.
- `gguf` backend: set `GEMMA_LABELING_BACKEND=gguf`, point `GEMMA_LABELING_MODEL_ID` at a local `.gguf` file, and set `GEMMA_LABELING_GGUF_MMPROJ_PATH` to the matching multimodal projector `.gguf`.

## Running with GGUF

GGUF is not loaded through Transformers. The app switches to a llama.cpp-backed runtime when `GEMMA_LABELING_MODEL_ID` ends in `.gguf` or when `GEMMA_LABELING_BACKEND=gguf`.

This app is multimodal, so a Gemma 4 GGUF needs both:

- the main model file, for example `gemma-4-E4B-it-Q4_K_M.gguf`
- the projector file, for example `mmproj-F16.gguf`

The GGUF backend talks to `llama-server` at `GEMMA_LABELING_LLAMA_SERVER_URL`. By default it can auto-start `llama-server` if `GEMMA_LABELING_LLAMA_SERVER_AUTO_START=1` and `GEMMA_LABELING_LLAMA_SERVER_BIN` is available on `PATH`.

Windows example:

```powershell
winget install llama.cpp
```

Example `.env` block for a local GGUF:

```env
GEMMA_LABELING_BACKEND=gguf
GEMMA_LABELING_MODEL_ID=C:\models\gemma-4-E4B-it-Q4_K_M.gguf
GEMMA_LABELING_GGUF_MMPROJ_PATH=C:\models\mmproj-F16.gguf
GEMMA_LABELING_LLAMA_SERVER_BIN=llama-server
GEMMA_LABELING_LLAMA_SERVER_AUTO_START=1
GEMMA_LABELING_LLAMA_SERVER_PORT=8081
GEMMA_LABELING_LLAMA_SERVER_N_GPU_LAYERS=999
```

In GGUF mode, the constrained path uses llama.cpp grammars one tag at a time: each step allows only the currently legal canonical tags plus `<END>`. That preserves no-repeat and unlock semantics without relying on Transformers logits processors.

## Project layout

- `data/taxonomy/e621_tags.yaml`: downloaded e621 snapshot used by default
- `src/gemma_labeling/e621_snapshot.py`: refreshes the local e621 snapshot
- `src/gemma_labeling/constraints.py`: token trie and constrained logits processor
- `src/gemma_labeling/model.py`: Gemma loader and generation wrapper
- `src/gemma_labeling/service.py`: prompt construction and response assembly
- `src/gemma_labeling/web.py`: FastAPI routes

## Refresh the e621 snapshot

```powershell
uv run python -m gemma_labeling.e621_snapshot
```

This downloads a local YAML snapshot from e621 using:
- top tags by category
- active tag aliases
- active tag implications

The default snapshot path can be overridden with `GEMMA_LABELING_TAXONOMY_PATH`.

## Manual demo checklist

- Upload an image that should only fit starter tags and compare baseline vs constrained output.
- Upload an image where `1girl` should unlock secondary tags like `solo`, `portrait`, or `long_hair`.
- Try an image where the baseline emits duplicates or unknown labels and confirm the constrained output stays canonical.
- Add a hint like `anime girl at sunset` and verify the prompt helps quality without breaking legality.

## Tests

```powershell
uv run pytest
```
