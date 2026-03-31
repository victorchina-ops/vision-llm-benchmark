# X (Twitter) Thread — Vision LLM Benchmark for Trail Camera Inference

---

**1/12**
🧵 Can open-source vision LLMs replace manual counting of hikers from trail cameras?

I tested 15+ local models + Claude Haiku/Sonnet/Opus on a real dataset from natural trail monitoring research — counting people, gender, age, backpacks & bicycles.

Full results 👇

---

**2/12**
**The context:**
We're part of a joint research project on monitoring visitors in open natural areas, in collaboration with the Israel Nature & Parks Authority, Ben-Gurion University, Technion & Ashkelon Academic College.

Trail cameras generate thousands of images. Manual review doesn't scale.

---

**3/12**
**The task:**
Each model receives a trail camera image and must return structured JSON:
- Total people count
- Gender per person (male/female/child)
- Backpack presence
- Bicycle presence

Scored against manual ground truth. People count = 50% of the score.

---

**4/12**
**The hardware:**
- RTX 3090 (24 GB) + RTX 3060 (12 GB) = 36 GB pooled VRAM
- All local models run fully on-device via Ollama — zero API cost, fully private
- Models tested at q4 / q8 / fp16 quantization where they fit in VRAM

---

**5/12**
**🥇 Winner: Qwen3-VL 8B @ q8 — score 91.5/100**

Alibaba's Qwen3 Vision-Language model. 8B parameters, 10 GB VRAM @ q8.
People count accuracy: 95.8% | Bicycle detection: 100%

Tiny model, huge accuracy. Runs in ~10s/image locally. This is the one to deploy.

---

**6/12**
**🥈 Qwen3-VL 30B MoE @ q8 — score 88.1/100**

Mixture-of-Experts: 30B total params but only ~3B active per inference.
Fits on dual-GPU (30 GB @ q8). Children detection: 100% — best of all models.

MoE architecture = large-model quality at mid-model speed. Impressive.

---

**7/12**
**🥉 Qwen3-VL 4B @ q8 — score 87.9/100**

Only 5 GB VRAM. Scores within 4 points of the 8B.
People count: 93% | Bicycle: 100%

If you only have a single mid-range GPU — this is your model.
The 4B/8B gap is surprisingly small for this structured task.

---

**8/12**
**Meta LLaMA 3.2 Vision 11B — score 85–87/100**

Meta's open vision model. Tested at q4 / q8 / fp16 (8–22 GB VRAM).
People count: 97–98% — highest of any model tested 🎯
Drops off on gender & backpack detail.

q4 vs fp16 difference: <3 points. Quantization barely hurts here.

---

**9/12**
**☁️ Claude Sonnet 4.6 (API) — score 85.6/100**

Strong people count (93%) and backpack detection (80.5%).
15s/image via API — slower than most local models.
No GPU required, but ongoing cost per inference.

Competitive with the best local models on this task.

---

**10/12**
**☁️ Claude Opus 4.6 — score 84.8 | Claude Haiku 4.5 — score 80.4**

Opus: great backpack detection (87%), but slower on children.
Haiku: fastest cloud option at 1.8s/image — great for high-volume pipelines where cost matters more than peak accuracy.

All three Claude models 100% bicycle detection.

---

**11/12**
**The rest of the field (exp1 — broad survey):**

| Model | Score |
|---|---|
| MiniCPM-V 8B | 80 |
| LLaVA 13B | 76 |
| Qwen2.5-VL 7B | 68 |
| Gemma3 12B | 67 |
| LLaVA 34B q4 | 71 |
| Moondream 1.8B | 20–46 |

Bigger ≠ better. Moondream (1.8B) collapsed at fp16/q8 — likely a tag/quantization issue.

---

**12/12**
**Key takeaways for trail camera inference:**

✅ Qwen3-VL 8B q8 is the sweet spot — 91.5 score, 10 GB VRAM, ~10s/image
✅ Qwen3-VL 4B works on a single mid-range GPU with minimal accuracy loss
✅ Quantization (q4→fp16) has surprisingly little impact on structured tasks
✅ Local models match or beat cloud APIs — and keep your data private
✅ All code & results open source 👇

github.com/victorchina-ops/vision-llm-benchmark

#ComputerVision #LLM #NatureMonitoring #OpenSource #Ollama #Qwen3
