# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Script

```bash
# Standard run — starts ollama if needed, sets GPU power limits, pulls missing models
python3 analyze_images.py

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

The entire project is a single script: `analyze_images.py`. There are no modules, packages, or tests.

**Key data structure:** `MODEL_VARIANTS` — a list of `(family, quant, ollama_tag, vram_gb)` tuples. This is the only place to add/remove/disable models. Entries that fail to pull are skipped gracefully; the rest of the run continues.

**Flow:**
1. `argparse` → parse `--gpu-power` / `--no-power-limit`
2. `set_gpu_power_limits()` → `sudo nvidia-smi -pl`
3. `ensure_ollama_running()` → HTTP check → `subprocess.Popen("ollama serve")` if needed
4. Pull loop → `pull_model()` per variant → builds `available[]` list of working variants
5. Analysis loop → `analyze_image()` per (variant, image) → `make_row()` → appends to `rows[]`
6. Save `runs/YYYY-MM-DD_HH-MM-SS/results.csv`, `results_raw.csv`, `benchmark.csv`
7. Three plot functions → saved to the same run folder

**Plotting functions:**
- `plot_model_quants()` — one PNG per model family; rows=images, cols=q4/q8/f16
- `plot_per_image_comparison()` — one PNG; rows=images, cols=models (best quant each)
- `plot_comparison()` — summary bar charts using best quant per model family

**`cell_text()`** is the single source of truth for what text appears in every stats cell across all plots. Edit here to change display format globally.

## Adding or Changing Models

Edit `MODEL_VARIANTS` in the config section. Check exact ollama tag names at `https://ollama.com/library/<model>/tags` — tag naming is inconsistent across models (e.g. `gemma3:12b-it-q4_K_M` vs `qwen2.5vl:7b-q4_K_M`). Use the scraping one-liner to verify:
```bash
curl -s https://ollama.com/library/<model>/tags | python3 -c "import sys,re; [print(t) for t in sorted(set(re.findall(r'[a-zA-Z0-9._:-]*(?:q4|q8|f16|fp16)[a-zA-Z0-9._:-]*', sys.stdin.read())))[:20]]"
```

## Planned Work

- **Ground truth evaluation**: user will provide manual classifications in `ground_truth.csv` (columns: `image, total_people, males, females, children, people_with_backpack, bicycle_present`). Script will be extended to compute precision/recall per model and quantization level.
