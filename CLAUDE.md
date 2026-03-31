# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scripts

There are two experiment scripts — both are fully self-contained (no shared modules):

| Script | Purpose |
|---|---|
| `analyze_images.py` | **Experiment 2** — current/recommended. 4 models at q8 only, with warm-up. |
| `analyze_images_exp1.py` | **Experiment 1** — broad survey. 6 model families at q4/q8/fp16. |

Both scripts share identical infrastructure (helpers, plotting, main loop). The only differences are `MODEL_VARIANTS` and the docstring.

## Running

```bash
# Experiment 2 (current)
python3 analyze_images.py

# Experiment 1 (broad survey)
python3 analyze_images_exp1.py

# Custom GPU power limits (GPU 0 then GPU 1, in watts)
python3 analyze_images.py --gpu-power 280 170

# Skip power limiting
python3 analyze_images.py --no-power-limit
```

Ollama must be reachable at `http://localhost:11434`. The script starts it automatically if not running, using `models/ollama/` as the model store. If ollama fails to start, check `/tmp/ollama_serve.log`.

If ollama was started manually outside the script, use:
```bash
OLLAMA_MODELS=/media/victor-rina/int_pixbackup/testllms/models/ollama ollama serve > /tmp/ollama_fixed.log 2>&1 &
```

## Hardware

- **GPU 0**: RTX 3090 24 GB — default power cap 250 W (set via `nvidia-smi -i 0 -pl 250`)
- **GPU 1**: RTX 3060 12 GB — 170 W (at max)
- **Total VRAM**: 36 GB pooled; ollama splits models across both GPUs automatically
- `~/.ollama/models` is a symlink → `models/ollama/` on this drive (7.3 TB)
- The `libggml-base.so.0` and CPU runner libs were manually copied to `/usr/local/lib/ollama/` and `libnvidia-ml.so` was symlinked there to fix GPU detection in ollama 0.19.0

## Architecture

**Key data structure:** `MODEL_VARIANTS` — a list of `(family, quant, ollama_tag, vram_gb)` tuples. This is the only place to add/remove/disable models. Entries that fail to pull are skipped gracefully; the rest of the run continues.

**Flow (both scripts):**
1. `argparse` → parse `--gpu-power` / `--no-power-limit`
2. `set_gpu_power_limits()` → `sudo nvidia-smi -pl`
3. `ensure_ollama_running()` → HTTP check → `subprocess.Popen("ollama serve")` if needed
4. Pull loop → `pull_model()` per variant → builds `available[]` list of working variants
5. Analysis loop → for each variant: `warmup_model()` (excluded from timing) → `analyze_image()` per image → `make_row()` → appends to `rows[]`
6. Incremental save: CSVs written after every variant; per-model plot saved as soon as all quants for that family complete
7. Final cross-model plots at end of run

**Plotting functions:**
- `plot_model_quants()` — one PNG per model family; rows=images, cols=quant levels
- `plot_per_image_comparison()` — one PNG; rows=images, cols=models (best quant each)
- `plot_comparison()` — summary bar charts using best quant per model family

**`cell_text()`** is the single source of truth for what text appears in every stats cell across all plots.

**`warmup_model()`** sends a 1×1 pixel image to force GPU loading before the benchmark loop. Timeout: 450 s. Errors are non-fatal.

## Adding or Changing Models

Edit `MODEL_VARIANTS` in the config section of the relevant script. Check exact ollama tag names — naming is inconsistent across models. Use this one-liner to verify:
```bash
curl -s https://ollama.com/library/<model>/tags | python3 -c "import sys,re; [print(t) for t in sorted(set(re.findall(r'[a-zA-Z0-9._:-]*(?:q4|q8|f16|fp16|bf16)[a-zA-Z0-9._:-]*', sys.stdin.read())))[:20]]"
```

Note: Qwen3-VL uses `bf16` tags (not `fp16`). Both are 2 bytes/param and labelled `f16` in plots.

## Planned Work

- **Ground truth evaluation**: user will provide manual classifications in `ground_truth.csv` (columns: `image, total_people, males, females, children, people_with_backpack, bicycle_present`). Script will be extended to compute precision/recall per model and quantization level.
