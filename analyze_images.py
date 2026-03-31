"""
Image analysis using Ollama multimodal LLMs.
Detects people (gender/age), backpacks, and bicycles in images.

For every model family, tests q4 / q8 / f16 (q16) variants where they fit in VRAM.
Outputs:
  results.csv            — every (model, quant, image) row
  benchmark.csv          — total / avg time per (model, quant)
  results_<model>.png    — per-model plot: images × quant levels side by side
  results_per_image.png  — per-image plot: all models (best quant) as columns
  results_comparison.png — summary bar charts across models (best quant each)

Requirements:
    pip install requests pandas matplotlib pillow
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_DIR  = Path("images")
RUNS_DIR    = Path("runs")          # each run saved under runs/YYYY-MM-DD_HH-MM-SS/
OLLAMA_URL  = "http://localhost:11434"

# Each entry: (family_label, quant_label, ollama_tag, vram_gb_estimate)
#
# Each entry: (family_label, quant_label, ollama_tag, vram_gb_estimate)
#
# VRAM budget: RTX 3090 (24 GB) + RTX 3060 (12 GB) = 36 GB pooled.
# fp16 = full 16-bit weights (~2 bytes/param).  q8 = ~1 byte/param.  q4 = ~0.5 byte/param.
# q4_K_M is preferred over plain q4_0 — better quality at same size.
# Models are skipped automatically at runtime if pull fails (tag not found / OOM).
MODEL_VARIANTS: list[tuple[str, str, str, float]] = [
    # ── Qwen2.5-VL 7B ─────────────────────────────────────────────────────────
    ("qwen2.5vl-7b",  "q4",  "qwen2.5vl:7b-q4_K_M",  6.0),
    ("qwen2.5vl-7b",  "q8",  "qwen2.5vl:7b-q8_0",    9.4),
    ("qwen2.5vl-7b",  "f16", "qwen2.5vl:7b-fp16",    15.0),

    # ── Qwen2.5-VL 3B (small/fast reference) ─────────────────────────────────
    ("qwen2.5vl-3b",  "q4",  "qwen2.5vl:3b-q4_K_M",  2.5),
    ("qwen2.5vl-3b",  "q8",  "qwen2.5vl:3b-q8_0",    4.0),
    ("qwen2.5vl-3b",  "f16", "qwen2.5vl:3b-fp16",    7.0),

    # ── LLaVA 13B v1.6 ────────────────────────────────────────────────────────
    ("llava-13b",     "q4",  "llava:13b-v1.6-vicuna-q4_K_M",  8.0),
    ("llava-13b",     "q8",  "llava:13b-v1.6-vicuna-q8_0",   16.0),
    ("llava-13b",     "f16", "llava:13b-v1.6-vicuna-fp16",   26.0),  # 3090 alone

    # ── Qwen2.5-VL 32B — best mid-large option for 36 GB ─────────────────────
    ("qwen2.5vl-32b", "q4",  "qwen2.5vl:32b-q4_K_M", 18.0),  # fits on 3090 alone
    ("qwen2.5vl-32b", "q8",  "qwen2.5vl:32b-q8_0",   34.0),  # split across both GPUs
    # f16 ~64 GB — exceeds budget, skipped

    # ── LLaVA 34B v1.6 ────────────────────────────────────────────────────────
    ("llava-34b",     "q4",  "llava:34b-v1.6-q4_K_M", 19.0),
    ("llava-34b",     "q8",  "llava:34b-v1.6-q8_0",   34.0),  # split across both GPUs
    # f16 ~68 GB — exceeds budget, skipped

    # ── Gemma 3 12B ───────────────────────────────────────────────────────────
    ("gemma3-12b",    "q4",  "gemma3:12b-it-q4_K_M",  8.0),
    ("gemma3-12b",    "q8",  "gemma3:12b-it-q8_0",   16.0),
    ("gemma3-12b",    "f16", "gemma3:12b-it-fp16",   24.0),  # fits on 3090 alone

    # ── Meta LLaMA 3.2 Vision 11B ─────────────────────────────────────────────
    ("llama3v-11b",   "q4",  "llama3.2-vision:11b-instruct-q4_K_M",  8.0),
    ("llama3v-11b",   "q8",  "llama3.2-vision:11b-instruct-q8_0",   12.0),
    ("llama3v-11b",   "f16", "llama3.2-vision:11b-instruct-fp16",   22.0),  # 3090 alone

    # ── Meta LLaMA 3.2 Vision 90B — too large even with RAM offload, skipped ──
    # Uncomment if you want to experiment (will be very slow, needs RAM offload):
    # ("llama3v-90b", "q4",  "llama3.2-vision:90b-instruct-q4_K_M", 48.0),

    # ── MiniCPM-V 8B v2.6 ─────────────────────────────────────────────────────
    ("minicpm-v-8b",  "q4",  "minicpm-v:8b-2.6-q4_K_M",  5.0),
    ("minicpm-v-8b",  "q8",  "minicpm-v:8b-2.6-q8_0",    9.0),
    ("minicpm-v-8b",  "f16", "minicpm-v:8b-2.6-fp16",    16.0),

    # ── Moondream 1.8B v2 ─────────────────────────────────────────────────────
    ("moondream",     "q4",  "moondream:1.8b-v2-q4_K_M",  1.5),
    ("moondream",     "q8",  "moondream:1.8b-v2-q8_0",    3.0),
    ("moondream",     "f16", "moondream:1.8b-v2-fp16",    6.0),
]

PROMPT = """\
Look at this image carefully. Count every person visible (including partial views).
Respond ONLY with a JSON object — no markdown fences, no extra text — in exactly this schema:

{
  "total_people": <integer>,
  "people": [
    {"gender": "male" | "female" | "child", "has_backpack": true | false}
  ],
  "bicycle_present": true | false
}

Rules:
- "child" = anyone appearing under ~16 years old.
- "has_backpack": only true if a backpack is clearly visible on that person.
- "bicycle_present": true if any bicycle (including parked) is in the frame.
- "total_people" must equal the length of the "people" list.
- If no people are visible, use an empty list and total_people = 0.
"""

QUANT_COLORS = {"q4": "#5c85d6", "q8": "#e07b39", "f16": "#4caf50"}
QUANT_ORDER  = ["q4", "q8", "f16"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(path: Path, max_px: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ensure_ollama_running() -> None:
    """Start ollama serve if it is not already running."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.ok:
            print("  [ok] ollama already running")
            return
    except requests.exceptions.ConnectionError:
        pass

    print("  ollama not running — starting server …", flush=True)
    env = {
        **__import__("os").environ,
        "OLLAMA_MODELS": str(Path("models/ollama").resolve()),
    }
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_serve.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    # wait up to 15 s for the server to be ready
    for _ in range(15):
        time.sleep(1)
        try:
            if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
                print("  [ok] ollama started")
                return
        except requests.exceptions.ConnectionError:
            pass
    sys.exit("ERROR: could not start ollama after 15 s — check /tmp/ollama_serve.log")


def installed_models() -> set[str]:
    try:
        out = subprocess.check_output(["ollama", "list"]).decode()
        return {line.split()[0] for line in out.splitlines()[1:] if line.strip()}
    except subprocess.CalledProcessError:
        return set()


def set_gpu_power_limits(limits: list[tuple[int, int]]) -> None:
    """Set power limit (watts) for each GPU index via nvidia-smi."""
    for gpu_idx, watts in limits:
        try:
            subprocess.run(
                ["sudo", "nvidia-smi", "-i", str(gpu_idx), "-pl", str(watts)],
                check=True, capture_output=True,
            )
            print(f"  GPU {gpu_idx}: power limit set to {watts}W")
        except subprocess.CalledProcessError as e:
            print(f"  GPU {gpu_idx}: failed to set power limit — {e.stderr.decode().strip()}")
            print("  (try running with sudo or set manually: sudo nvidia-smi -i N -pl W)")


def pull_model(tag: str) -> bool:
    """Pull model. Returns True on success, False on failure."""
    if tag in installed_models():
        print(f"  [ok] {tag} already installed")
        return True
    print(f"  Pulling {tag} …", flush=True)
    proc = subprocess.Popen(
        ["ollama", "pull", tag],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(f"    {line.rstrip()}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"  [SKIP] pull failed for {tag} (exit {proc.returncode})")
        return False
    print(f"  [ok] {tag} ready")
    return True


def analyze_image(tag: str, image_path: Path) -> tuple[dict | None, str, float]:
    payload = {
        "model": tag,
        "prompt": PROMPT,
        "images": [encode_image(image_path)],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        elapsed = time.time() - t0
        raw = resp.json().get("response", "")
        return json.loads(raw), raw, elapsed
    except Exception as exc:
        return None, str(exc), time.time() - t0


def make_row(family: str, quant: str, tag: str,
             img_name: str, data: dict | None, raw: str, elapsed: float) -> dict:
    row = {
        "family": family, "quant": quant, "model_tag": tag,
        "image": img_name, "elapsed_sec": round(elapsed, 1),
        "total_people": 0, "males": 0, "females": 0, "children": 0,
        "people_with_backpack": 0, "bicycle_present": False,
        "parse_error": data is None, "raw_response": raw,
    }
    if data:
        people = data.get("people", [])
        row["total_people"]         = data.get("total_people", len(people))
        row["males"]                = sum(1 for p in people if p.get("gender") == "male")
        row["females"]              = sum(1 for p in people if p.get("gender") == "female")
        row["children"]             = sum(1 for p in people if p.get("gender") == "child")
        row["people_with_backpack"] = sum(1 for p in people if p.get("has_backpack"))
        row["bicycle_present"]      = bool(data.get("bicycle_present", False))
    return row


def cell_text(r: pd.Series | None) -> tuple[str, str, str]:
    """Returns (text, fg_color, bg_color) for a stats cell."""
    if r is None or r["parse_error"]:
        return "ERROR", "red", "#ffebee"
    bike = "yes" if r["bicycle_present"] else "no"
    txt = (
        f"people: {int(r['total_people'])}\n"
        f"male: {int(r['males'])}\n"
        f"female: {int(r['females'])}\n"
        f"child: {int(r['children'])}\n"
        f"backpack: {int(r['people_with_backpack'])}\n"
        f"bicycle: {bike}\n"
        f"time: {r['elapsed_sec']}s"
    )
    return txt, "black", "#e8f5e9" if r["total_people"] > 0 else "#fafafa"


# ── Plot 1: per-model quantization comparison ─────────────────────────────────
# Rows = images, Cols = [ref image] + [q4 stats] + [q8 stats] + [f16 stats]

def plot_model_quants(family: str, quants_available: list[str],
                      df: pd.DataFrame, images: list[Path],
                      out_dir: Path) -> None:
    n_imgs  = len(images)
    n_quant = len(quants_available)
    col_w   = [2.5] + [1.5] * n_quant

    fig, axes = plt.subplots(
        n_imgs, 1 + n_quant,
        figsize=(sum(col_w) * 1.1, 3.8 * n_imgs),
        gridspec_kw={"width_ratios": col_w},
    )
    if n_imgs == 1:
        axes = [axes]

    fig.suptitle(f"{family} — Quantization Comparison", fontsize=13,
                 fontweight="bold", y=1.001)

    axes[0][0].set_title("Image", fontsize=9, fontweight="bold")
    for j, q in enumerate(quants_available):
        axes[0][j + 1].set_title(q.upper(), fontsize=10, fontweight="bold",
                                  color=QUANT_COLORS.get(q, "#333"))

    for i, img_path in enumerate(images):
        ax_img = axes[i][0]
        try:
            ax_img.imshow(mpimg.imread(str(img_path)))
        except Exception:
            ax_img.text(0.5, 0.5, "load error", ha="center", va="center",
                        transform=ax_img.transAxes)
        ax_img.set_ylabel(img_path.name, fontsize=7, rotation=0,
                          labelpad=4, va="center", ha="right")
        ax_img.axis("off")

        for j, q in enumerate(quants_available):
            ax = axes[i][j + 1]
            ax.axis("off")
            sub = df[(df["family"] == family) & (df["quant"] == q) &
                     (df["image"] == img_path.name)]
            r = sub.iloc[0] if not sub.empty else None
            txt, fg, bg = cell_text(r)
            ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                    fontsize=9, va="center", ha="center",
                    fontfamily="monospace", color=fg,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=bg,
                              edgecolor=QUANT_COLORS.get(q, "#ccc"), linewidth=1.5,
                              alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.999])
    safe = family.replace(":", "_").replace("/", "_")
    out  = out_dir / f"results_{safe}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: per-image comparison across models (best quant each) ──────────────

def plot_per_image_comparison(df: pd.DataFrame, images: list[Path],
                               families: list[str], out_dir: Path) -> None:
    # For each family pick highest available quant in QUANT_ORDER
    best: dict[str, str] = {}
    for fam in families:
        for q in reversed(QUANT_ORDER):
            if not df[(df["family"] == fam) & (df["quant"] == q)].empty:
                best[fam] = q
                break

    n_imgs   = len(images)
    n_models = len(families)
    col_w    = [2.5] + [1.4] * n_models

    fig, axes = plt.subplots(
        n_imgs, 1 + n_models,
        figsize=(sum(col_w) * 1.05, 3.8 * n_imgs),
        gridspec_kw={"width_ratios": col_w},
    )
    if n_imgs == 1:
        axes = [axes]

    fig.suptitle("All Models — Per-Image Comparison (best quant per model)",
                 fontsize=13, fontweight="bold", y=1.001)

    axes[0][0].set_title("Image", fontsize=9, fontweight="bold")
    for j, fam in enumerate(families):
        q = best.get(fam, "q4")
        axes[0][j + 1].set_title(f"{fam}\n({q.upper()})", fontsize=7,
                                   fontweight="bold")

    for i, img_path in enumerate(images):
        ax_img = axes[i][0]
        try:
            ax_img.imshow(mpimg.imread(str(img_path)))
        except Exception:
            ax_img.text(0.5, 0.5, "load error", ha="center", va="center",
                        transform=ax_img.transAxes)
        ax_img.set_ylabel(img_path.name, fontsize=7, rotation=0,
                          labelpad=4, va="center", ha="right")
        ax_img.axis("off")

        for j, fam in enumerate(families):
            ax = axes[i][j + 1]
            ax.axis("off")
            q   = best.get(fam, "q4")
            sub = df[(df["family"] == fam) & (df["quant"] == q) &
                     (df["image"] == img_path.name)]
            r   = sub.iloc[0] if not sub.empty else None
            txt, fg, bg = cell_text(r)
            ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                    fontsize=8, va="center", ha="center",
                    fontfamily="monospace", color=fg,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                              edgecolor="#cccccc", alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.999])
    fig.savefig(out_dir / "results_per_image.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'results_per_image.png'}")


# ── Plot 3: summary comparison (best quant per model) ─────────────────────────

def plot_comparison(df: pd.DataFrame, benchmark: dict,
                    families: list[str], out_dir: Path) -> None:
    # pick best quant per family
    best_rows = []
    for fam in families:
        for q in reversed(QUANT_ORDER):
            sub = df[(df["family"] == fam) & (df["quant"] == q) &
                     (~df["parse_error"])]
            if not sub.empty:
                best_rows.append(sub.copy().assign(label=f"{fam}\n({q.upper()})"))
                break
    if not best_rows:
        print("  No data for comparison plot.")
        return

    ok = pd.concat(best_rows)

    metrics = [
        ("total_people",        "Avg people detected",      "#5c85d6"),
        ("elapsed_sec",         "Avg time/image (s)",       "#e07b39"),
        ("people_with_backpack","Avg backpack detections",  "#4caf50"),
        ("bicycle_present",     "Bicycle detections (sum)", "#9c27b0"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle("Model Comparison — Best Quantization per Model",
                 fontsize=13, fontweight="bold")

    for ax, (col, title, color) in zip(axes[:4], metrics):
        agg  = ok.groupby("label")[col]
        vals = agg.sum() if col == "bicycle_present" else agg.mean()
        ax.bar(range(len(vals)), vals.values, color=color, edgecolor="white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        for j, v in enumerate(vals.values):
            ax.text(j, v + 0.01 * max(vals.values, default=1),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # benchmark panel
    ax_b   = axes[4]
    labels, totals, avgs = [], [], []
    for fam in families:
        for q in reversed(QUANT_ORDER):
            key = (fam, q)
            if key in benchmark:
                labels.append(f"{fam}\n({q.upper()})")
                totals.append(benchmark[key]["total_sec"])
                avgs.append(benchmark[key]["avg_sec"])
                break

    x = range(len(labels))
    ax_b.bar(x, totals, color="#ff7043", edgecolor="white", label="Total (s)")
    ax_b.plot(x, avgs, "o--", color="#333", linewidth=1.5,
              markersize=5, label="Avg/img (s)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax_b.set_title("Benchmark: total time (s)", fontsize=10)
    ax_b.legend(fontsize=8)
    for j, t in enumerate(totals):
        ax_b.text(j, t + 0.5, f"{t}s", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "results_comparison.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'results_comparison.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal LLM image analysis")
    parser.add_argument(
        "--gpu-power", type=int, nargs="+", metavar="W",
        default=[250, 170],
        help="Power limits in watts for GPU 0, GPU 1, … (default: 250 170)",
    )
    parser.add_argument(
        "--no-power-limit", action="store_true",
        help="Skip setting GPU power limits",
    )
    args = parser.parse_args()

    # ── Set GPU power limits ──────────────────────────────────────────────────
    if not args.no_power_limit:
        print("=== Setting GPU power limits ===")
        limits = list(enumerate(args.gpu_power))
        set_gpu_power_limits(limits)
        print()

    # ── Ensure ollama is running ──────────────────────────────────────────────
    print("=== Checking ollama ===")
    ensure_ollama_running()
    print()

    images = sorted(IMAGES_DIR.glob("*.[Jj][Pp][Gg]")) + \
             sorted(IMAGES_DIR.glob("*.[Pp][Nn][Gg]"))
    if not images:
        sys.exit(f"No images found in {IMAGES_DIR}/")
    print(f"Found {len(images)} image(s) in {IMAGES_DIR}/\n")

    # ── Create timestamped output folder ─────────────────────────────────────
    run_ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RUNS_DIR / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    # ── 1. Pull all variants ──────────────────────────────────────────────────
    print("=== Pulling models ===")
    available: list[tuple[str, str, str]] = []   # (family, quant, tag) that pulled OK
    for family, quant, tag, _vram in MODEL_VARIANTS:
        if pull_model(tag):
            available.append((family, quant, tag))
    print(f"\n{len(available)}/{len(MODEL_VARIANTS)} variants available\n")

    if not available:
        sys.exit("No models available.")

    # ── 2. Analyse ────────────────────────────────────────────────────────────
    print("=== Analysing images ===")
    rows      = []
    benchmark = {}   # (family, quant) → {total_sec, avg_sec, errors}

    for family, quant, tag in available:
        print(f"\n[{family} / {quant.upper()}]  ({tag})")
        t0     = time.time()
        errors = 0
        for img in images:
            print(f"  {img.name} … ", end="", flush=True)
            data, raw, elapsed = analyze_image(tag, img)
            row = make_row(family, quant, tag, img.name, data, raw, elapsed)
            rows.append(row)
            if row["parse_error"]:
                errors += 1
                print(f"ERROR ({elapsed:.1f}s)")
            else:
                print(
                    f"{row['total_people']} people "
                    f"(M={row['males']} F={row['females']} C={row['children']})  "
                    f"packs={row['people_with_backpack']}  "
                    f"bike={'Y' if row['bicycle_present'] else 'N'}  "
                    f"{elapsed:.1f}s"
                )
        total = time.time() - t0
        benchmark[(family, quant)] = {
            "total_sec": round(total, 1),
            "avg_sec":   round(total / len(images), 1),
            "errors":    errors,
        }
        print(f"  ── total: {total:.1f}s  avg/img: {total/len(images):.1f}s  errors: {errors}")

    # ── 3. Save CSVs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    clean_cols = ["family", "quant", "model_tag", "image", "total_people", "males",
                  "females", "children", "people_with_backpack", "bicycle_present",
                  "elapsed_sec", "parse_error"]
    df[clean_cols].to_csv(out_dir / "results.csv", index=False)
    print(f"\nSaved CSV: {out_dir / 'results.csv'}")

    # raw responses in a separate file to keep results.csv clean
    df[["family", "quant", "model_tag", "image", "raw_response"]].to_csv(
        out_dir / "results_raw.csv", index=False
    )
    print(f"Saved CSV: {out_dir / 'results_raw.csv'}")

    bdf = pd.DataFrame([
        {"family": f, "quant": q, **v} for (f, q), v in benchmark.items()
    ])
    bdf.to_csv(out_dir / "benchmark.csv", index=False)
    print(f"Saved CSV: {out_dir / 'benchmark.csv'}")
    print("\n=== Benchmark Summary ===")
    print(bdf.to_string(index=False))

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    families_done = list(dict.fromkeys(f for f, _, _ in available))  # ordered unique

    # Plot A: per-model, rows=images, cols=quant levels
    for fam in families_done:
        quants = [q for f, q, _ in available if f == fam]
        quants_ordered = [q for q in QUANT_ORDER if q in quants]
        plot_model_quants(fam, quants_ordered, df, images, out_dir)

    # Plot B: per-image, cols = one model each (best quant)
    plot_per_image_comparison(df, images, families_done, out_dir)

    # Plot C: summary bar charts
    plot_comparison(df, benchmark, families_done, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
