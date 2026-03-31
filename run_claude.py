"""
Run the image analysis benchmark using any Claude model via the Anthropic API.
Results saved to runs/claude_<model_shortname>/ in the same CSV format as local runs.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 run_claude.py --model haiku      # claude-haiku-4-5-20251001
    python3 run_claude.py --model sonnet     # claude-sonnet-4-6
    python3 run_claude.py --model opus       # claude-opus-4-6

Requirements:
    pip install anthropic pillow pandas
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
from PIL import Image

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_DIR = Path("images")

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


def analyze_image(client: anthropic.Anthropic, model_id: str,
                  image_path: Path) -> tuple[dict | None, str, float]:
    b64 = encode_image(image_path)
    t0 = time.time()
    try:
        msg = client.messages.create(
            model=model_id,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        elapsed = time.time() - t0
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw), raw, elapsed
    except Exception as exc:
        return None, str(exc), time.time() - t0


def make_row(family: str, model_id: str, img_name: str,
             data: dict | None, raw: str, elapsed: float) -> dict:
    row = {
        "family":               family,
        "quant":                "api",
        "model_tag":            model_id,
        "image":                img_name,
        "elapsed_sec":          round(elapsed, 1),
        "total_people":         0,
        "males":                0,
        "females":              0,
        "children":             0,
        "people_with_backpack": 0,
        "bicycle_present":      False,
        "parse_error":          data is None,
        "raw_response":         raw,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Claude API image benchmark")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="haiku",
        help="Which Claude model to run (default: haiku)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY environment variable before running.")

    model_id  = MODELS[args.model]
    family    = f"claude-{args.model}"
    out_dir   = Path(f"runs/claude_{args.model}")

    images = sorted(IMAGES_DIR.glob("*.[Jj][Pp][Gg]")) + \
             sorted(IMAGES_DIR.glob("*.[Pp][Nn][Gg]"))
    if not images:
        sys.exit(f"No images found in {IMAGES_DIR}/")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Model : {model_id}")
    print(f"Images: {len(images)}")
    print(f"Output: {out_dir}\n")

    rows    = []
    errors  = 0
    t_total = time.time()

    for img in images:
        print(f"  {img.name} … ", end="", flush=True)
        data, raw, elapsed = analyze_image(client, model_id, img)
        row = make_row(family, model_id, img.name, data, raw, elapsed)
        rows.append(row)
        if row["parse_error"]:
            errors += 1
            print(f"ERROR ({elapsed:.1f}s)  {raw[:80]}")
        else:
            print(
                f"{row['total_people']} people "
                f"(M={row['males']} F={row['females']} C={row['children']})  "
                f"packs={row['people_with_backpack']}  "
                f"bike={'Y' if row['bicycle_present'] else 'N'}  "
                f"{elapsed:.1f}s"
            )

    total_elapsed = time.time() - t_total
    print(f"\nTotal: {total_elapsed:.1f}s  avg/img: {total_elapsed/len(images):.1f}s  errors: {errors}")

    clean_cols = ["family", "quant", "model_tag", "image", "total_people", "males",
                  "females", "children", "people_with_backpack", "bicycle_present",
                  "elapsed_sec", "parse_error"]
    df = pd.DataFrame(rows)
    df[clean_cols].to_csv(out_dir / "results.csv", index=False)
    df[["family", "quant", "model_tag", "image", "raw_response"]].to_csv(
        out_dir / "results_raw.csv", index=False
    )
    pd.DataFrame([{
        "family": family, "quant": "api",
        "total_sec": round(total_elapsed, 1),
        "avg_sec":   round(total_elapsed / len(images), 1),
        "errors":    errors,
    }]).to_csv(out_dir / "benchmark.csv", index=False)
    print(f"Saved: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
