# Multimodal LLM Image Analysis Benchmark

Automated benchmark that runs vision LLMs via [Ollama](https://ollama.com) on a set of images, detecting people (gender/age), backpacks, and bicycles. Results are saved per-run in timestamped folders with CSV tables and comparison plots.

Two experiments are included — each is a fully self-contained script.

---

## Hardware Configuration

| Component | Spec |
|---|---|
| GPU 0 | NVIDIA RTX 3090 — 24 GB VRAM |
| GPU 1 | NVIDIA RTX 3060 — 12 GB VRAM |
| Total VRAM | 36 GB pooled |
| CPU | Intel i7-7740X @ 4.30 GHz |
| RAM | 64 GB |
| OS | Linux (Ubuntu) |
| Driver | 570.211.01 |

The RTX 3090 is capped at **250W** by default (down from 350W) since LLM inference is memory-bandwidth bound. Override with `--gpu-power`.

---

## Experiments

### Experiment 1 — Broad model survey (`analyze_images_exp1.py`)

Wide comparison across 6 model families tested at q4, q8, and fp16 quantizations.

| Family | Sizes | q4 | q8 | fp16 | Notes |
|---|---|---|---|---|---|
| `qwen2.5vl` | 3B, 7B, 32B | ✓ | ✓ | ✓ (3B/7B) | Qwen2.5 Vision-Language |
| `llava` | 13B, 34B | ✓ | ✓ | ✓ (13B) | LLaVA v1.6 Vicuna base |
| `gemma3` | 12B | ✓ | ✓ | ✓ | Google Gemma 3 multimodal |
| `llama3.2-vision` | 11B | ✓ | ✓ | ✓ | Meta LLaMA 3.2 Vision |
| `minicpm-v` | 8B | ✓ | ✓ | ✓ | MiniCPM-V 2.6 |
| `moondream` | 1.8B | ✓ | ✓ | ✓ | Ultra-fast reference |

---

### Experiment 2 — Focused benchmark (`analyze_images.py`)

Focused comparison of the best-performing / most promising models at q8 only. Includes a warm-up step so model load time is excluded from benchmark timings.

| Family | Size | q8 | Notes |
|---|---|---|---|
| `llama3.2-vision` | 11B | ✓ | Meta LLaMA 3.2 Vision |
| `qwen3-vl` | 4B | ✓ | Qwen3 Vision-Language (bf16) |
| `qwen3-vl` | 8B | ✓ | Qwen3 Vision-Language (bf16) |
| `qwen3-vl` | 30B-A3B MoE | ✓ | 30B total / ~3B active; split across both GPUs |

VRAM budget per quantization (approximate):
- **q4_K_M** — 0.5 bytes/param
- **q8_0** — 1 byte/param
- **fp16/bf16** — 2 bytes/param (Qwen3-VL uses bfloat16)

Models are skipped automatically at runtime if pull fails.

---

## What It Detects

For every image the models answer:

| Field | Type | Description |
|---|---|---|
| `total_people` | integer | Total visible persons including partial views |
| `males` | integer | Adults identified as male |
| `females` | integer | Adults identified as female |
| `children` | integer | Persons appearing under ~16 years old |
| `people_with_backpack` | integer | Persons with a clearly visible backpack |
| `bicycle_present` | boolean | Any bicycle visible (including parked) |
| `elapsed_sec` | float | Inference time for that image (excludes load time) |

---

## Project Structure

```
testllms/
├── analyze_images.py           # Experiment 2 (current / recommended)
├── analyze_images_exp1.py      # Experiment 1 (broad survey)
├── images/                     # input images (JPG/PNG)
├── models/
│   └── ollama/                 # downloaded model weights
└── runs/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── results.csv             # clean per-(model, quant, image) table
        ├── results_raw.csv         # raw JSON responses
        ├── benchmark.csv           # total & avg inference time per variant
        ├── results_<family>.png    # per-model: images × quant levels
        ├── results_per_image.png   # per-image: all models side by side
        └── results_comparison.png  # summary bar charts
```

---

## Usage

```bash
# Experiment 2 — recommended starting point
python3 analyze_images.py

# Experiment 1 — broad survey across more models and quant levels
python3 analyze_images_exp1.py

# Custom GPU power limits (watts, GPU 0 then GPU 1)
python3 analyze_images.py --gpu-power 280 170

# Skip power limiting
python3 analyze_images.py --no-power-limit
```

### Requirements

```bash
pip install requests pandas matplotlib pillow
```

Ollama is started automatically if not already running. Models are pulled on first use.

---

## Output Plots

### `results_<family>.png` — Per-model quantization comparison
One plot per model family. Rows = images, columns = quant levels side by side. Shows how quantization affects accuracy for the same model.

### `results_per_image.png` — Cross-model comparison
One row per image, one column per model (best available quantization). Easiest way to compare how different models see the same scene.

### `results_comparison.png` — Summary bar charts
Aggregated metrics across all models: avg people detected, avg inference time, backpack detections, bicycle detections, and total benchmark time.

---

## Planned: Manual Ground Truth Evaluation

A manual classification of each sample image will be added to allow precision/recall scoring per model and quantization level.

Planned metrics:
- **People count accuracy** — absolute error vs ground truth
- **Gender classification accuracy** — per-person correct/incorrect
- **Backpack detection** — precision / recall
- **Bicycle detection** — precision / recall
- **Speed vs accuracy trade-off** — plot per model family

Ground truth file format (to be added): `ground_truth.csv`

```
image,total_people,males,females,children,people_with_backpack,bicycle_present
IMAG0019.JPG,6,4,2,0,1,False
...
```

---

## Notes

- Images are resized to max 1024px on the longest side before inference to avoid timeouts
- All models use `temperature=0` for deterministic outputs
- `format: json` is passed to Ollama to enforce structured output
- Each run is saved in a timestamped folder — old runs are never overwritten
- The `~/.ollama/models` symlink points to `models/ollama/` on this drive to avoid filling the system disk
- Warm-up requests (1×1 pixel image) are sent before each model's benchmark loop — load time is logged separately and excluded from per-image timings
