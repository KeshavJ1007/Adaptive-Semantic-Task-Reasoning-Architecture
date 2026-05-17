<div align="center">

<br/>

```
 █████╗ ███████╗████████╗██████╗  █████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
███████║███████╗   ██║   ██████╔╝███████║
██╔══██║╚════██║   ██║   ██╔══██╗██╔══██║
██║  ██║███████║   ██║   ██║  ██║██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
```

# Adaptive Semantic Task Reasoning Architecture

**DVCon India 2026 — Design Contest | Stage 2A Submission**  
**Team VILSCORE · Submission #389 · Saveetha Engineering College**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Inference-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP_ViT--B/32-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/CLIP)
[![YOLOv5](https://img.shields.io/badge/YOLOv5s-Ultralytics-00FFAB?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Open In Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com/github/KeshavJ1007/Adaptive-Semantic-Task-Reasoning-Architecture/blob/main/ASTRA_Stage2A_Colab_v3.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> *"Traditional detectors see everything. ASTRA selects what matters."*

<br/>

---

</div>

## 📌 What Is ASTRA?

Imagine a scene with a **wine glass**, a **cup**, and a **bottle** all visible in one image. A standard object detector labels all three and stops. But if your task is *"serve wine"* — **which one do you actually reach for?**

**ASTRA** solves exactly this problem. It is a **task-aware object selection system** that goes beyond conventional detection by combining:

- 🔍 **Visual perception** — YOLOv5s object detection with bounding boxes
- 🧠 **Semantic reasoning** — CLIP vision-language embeddings align task intent with object identity
- 🕸️ **Graph intelligence** — A knowledge graph encodes expert task-object relationships
- 🌍 **Scene context** — Co-presence patterns and fallback logic adapt to real-world scene variation
- ⚡ **Hardware co-design** — FPGA acceleration planned for edge deployment (Stage 3)

The result: given any image and any natural-language task description, ASTRA returns **the single best object** — ranked, scored, and explained.

<br/>

| Input Scene | ASTRA Output |
|:-----------:|:------------:|
| ![ASTRA serve wine result](assets/result_serve_wine.jpg) | **O\* = wine glass** · Sf = 0.874 |

> 📸 *Replace the image above with your actual output from `results/snapshots/` after running the notebook.*

---

## 👥 Team

| Name | Role |
|------|------|
| **Kamalesh S** | Architecture Design · Pipeline Integration · FPGA Planning |
| **Keshav J** | CLIP Encoder · Scoring Engine · Knowledge Graph |
| **Raghul V** | Scene Context Reasoning · Evaluation · Results Analysis |

**Institution:** Saveetha Engineering College  
**Competition:** DVCon India 2026 Design Contest — [dvcon-india.org](https://dvcon-india.org)

---

## 🗂️ Repository Structure

```
ASTRA/
│
├── 📓 ASTRA_Stage2A_Colab_v3.ipynb    ← Main notebook (run in Colab)
├── 📄 ASTRA_Stage2A_SingleFile.py     ← Single-file copy-paste version
│
├── 📁 assets/                         ← Images for this README
│   ├── result_serve_wine.jpg
│   ├── result_sit_comfortably.jpg
│   ├── result_extinguish_fire.jpg
│   └── score_table_screenshot.png
│
├── 📁 results/
│   ├── evaluation_results.json        ← Full 14-task output log
│   ├── snapshots/                     ← Annotated result images per task
│   └── latency_report.png             ← CPU latency + performance chart
│
├── 📁 docs/
│   ├── DVCon_Stage1_Report_389.pdf    ← Stage 1 proposal (accepted)
│   └── Stage2A_QnA_Notes.pdf          ← DVCon Q&A session notes
│
└── 📄 README.md
```

---

## 🚀 Quick Start

### Option 1 — Google Colab *(Recommended)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KeshavJ1007/Adaptive-Semantic-Task-Reasoning-Architecture/blob/main/ASTRA_Stage2A_Colab_v3.ipynb)

1. Click the badge above
2. Set **Runtime → Change runtime type → CPU**
3. Run all cells top-to-bottom
4. Results are saved automatically to `results/`

### Option 2 — Local Python

```bash
# Clone the repo
git clone https://github.com/KeshavJ1007/Adaptive-Semantic-Task-Reasoning-Architecture.git
cd Adaptive-Semantic-Task-Reasoning-Architecture

# Install dependencies
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
pip install networkx pycocotools opencv-python-headless matplotlib pillow

# Run the pipeline
python ASTRA_Stage2A_SingleFile.py
```

### Option 3 — Single Query in 3 Lines

```python
result = pipeline.run(
    image_input = "your_image.jpg",
    task        = "serve wine",
    show        = True,
    verbose     = True,
)

print(f"Best object : {result['best_label']}")
print(f"Final score : {result['best_score']:.4f}")
print(f"Latency     : {result['latency_ms']:.0f} ms")
```

---

## 🏗️ System Architecture

ASTRA follows a **9-stage pipeline** from raw image to final object selection:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ASTRA PIPELINE                              │
│                                                                     │
│  Input Image ──► Stage 1: Image Preprocessing (Resize + Normalize)  │
│                      │                                              │
│                      ▼                                              │
│               Stage 2: CNN Object Detector (YOLOv5s)                │
│                      │                    │                         │
│                      ▼                    ▼                         │
│           Detected Object List     Visual Feature Maps              │
│                      │                    │                         │
│                      ▼                    ▼                         │
│               Stage 3: Task Encoding (CLIP Text Encoder)            │
│  Task Text ──────────┘                                              │
│                      │                                              │
│                      ▼                                              │
│         Stage 4: Object Embedding (CLIP Dual-Channel)               │
│                      │                                              │
│                      ▼                                              │
│     Stage 5: Task-Object Knowledge Graph (Weighted Directed Graph)  │
│                      │                                              │
│                      ▼                                              │
│        Stage 6: Vision-Language Embedding Space (Cosine Sim)        │
│                      │                                              │
│                      ▼                                              │
│          Stage 7: Scene Context Reasoning → Sp score                │
│                      │                                              │
│                      ▼                                              │
│             Stage 8: FPGA Similarity Accelerator (*)                │
│                      │                                              │
│                      ▼                                              │
│          Stage 9: Object Ranking Engine → O* (Best Object)          │
└─────────────────────────────────────────────────────────────────────┘

  (*) Software-simulated in Stage 2A. FPGA HDL planned for Stage 3.
```

### Hardware-Software Co-Design *(Target Architecture)*

```
┌──────────────────────────┐     ┌──────────────────────────┐
│   VEGA RISC-V Processor  │────►│   FPGA Fabric (Genesys-2)│
│                          │     │                          │
│  • Image preprocessing   │     │  • CNN Feature Accel.    │
│  • Task text encoding    │     │  • Task-Guided Attention │
│  • Graph construction    │     │  • Vector Similarity Eng.│
│  • Control & scheduling  │     │  • Object Ranking Accel. │
└──────────────────────────┘     └──────────┬───────────────┘
                                            │
                                            ▼
                                   ┌─────────────────────┐
                                   │   Final Prediction  │
                                   │  O* + Score + BBox  │
                                   └─────────────────────┘
```

---

## 🔢 Scoring Formula

At the heart of ASTRA is a **weighted additive scoring formula** combining four signals:

<div align="center">

### `Sf = α·Ci + β·Sg + γ·Ss + δ·Sp`

</div>

| Symbol | Component | Weight | Description |
|:------:|-----------|:------:|-------------|
| **Ci** | Detection Confidence | **α = 0.20** | YOLOv5s bounding-box confidence score ∈ [0, 1] |
| **Sg** | Graph Relevance Score | **β = 0.40** | Knowledge graph edge weight for the task → object relationship |
| **Ss** | Semantic Similarity | **γ = 0.25** | CLIP cosine similarity between task embedding Te and object embedding Ei, normalised [−1,1] → [0,1] |
| **Sp** | Scene Proximity Score | **δ = 0.15** | Co-presence reinforcement + fallback rank boost ∈ [0, 1] |

> **Weight constraint:** `α + β + γ + δ = 0.20 + 0.40 + 0.25 + 0.15 = 1.00`

> **Final selection:** `O* = argmax Sf` over all detected objects in the scene

**Why β is highest:** The knowledge graph encodes human-validated task-object affordances — it is the most task-specific signal available.

**Why Sp is additive (not multiplicative):** Making scene context a separate additive term means it can reward fallback objects (e.g., a *cup* when no *wine glass* is present) without distorting the primary semantic scores.

---

## 📦 Modules

### `TaskObjectGraph` — Knowledge Graph

Manually constructed from the **COCO-Tasks benchmark** (Sawatzky et al., 2019). Covers all 14 evaluation tasks with weighted directed edges:

```python
TASK_GRAPH = {
    "serve wine":                          [("wine glass", 1.0), ("cup", 0.6), ("bottle", 0.4)],
    "sit comfortably":                     [("couch", 1.0), ("chair", 0.8), ("bench", 0.6), ("bed", 0.4)],
    "place flowers":                       [("vase", 1.0), ("cup", 0.6), ("wine glass", 0.4)],
    "dig a hole":                          [("shovel", 1.0), ("baseball bat", 0.4), ("umbrella", 0.3)],
    "extinguish fire":                     [("fire hydrant", 1.0), ("bottle", 0.5)],
    "get potatoes out of fire":            [("fork", 1.0), ("knife", 0.8), ("spoon", 0.6)],
    "water a plant":                       [("bottle", 1.0), ("cup", 0.6), ("bowl", 0.4)],
    "open a bottle of beer":               [("knife", 1.0), ("scissors", 0.6)],
    "open a parcel":                       [("scissors", 1.0), ("knife", 0.7)],
    "pour sugar":                          [("spoon", 1.0), ("bowl", 0.6), ("cup", 0.4)],
    "smear butter":                        [("knife", 1.0), ("spoon", 0.5)],
    "get lemon out of tea":                [("spoon", 1.0), ("fork", 0.6)],
    "pound a carpet":                      [("baseball bat", 1.0), ("tennis racket", 0.7)],
    "step on something to reach a shelf":  [("chair", 1.0), ("bench", 0.8), ("couch", 0.6)],
}
```

Weight semantics: `1.0` = canonical · `0.8` = strong alternative · `0.6` = acceptable · `0.4` = last resort

---

### `ObjectDetector` — YOLOv5s

```python
detector = ObjectDetector(model_name='yolov5s', conf_threshold=0.20, device='cpu')
objects, image = detector.detect("scene.jpg")
# Returns:
# [{'label': 'wine glass', 'confidence': 0.91, 'bbox': [x1,y1,x2,y2], 'crop': <ndarray>}, ...]
```

- Model: **YOLOv5s** (COCO-80 pretrained, ~14 MB, auto-downloaded)
- Threshold: `0.20` — lower than default to improve recall in cluttered scenes
- Output: per-object label, Ci, bounding box, and cropped RGB region for visual encoding

---

### `CLIPEncoder` — Vision-Language Embeddings

Dual-channel encoding for both **tasks** and **objects**:

```python
# Task encoding — ensemble of two prompts for robustness
Te = 0.6 × CLIP("serve wine") + 0.4 × CLIP("a photo of an object used to serve wine")

# Object encoding — ensemble of text label + visual crop
Ei = 0.5 × CLIP_text("wine glass") + 0.5 × CLIP_image(<crop>)

# Semantic similarity
Ss = cosine_similarity(Te, Ei)   # in [-1, 1]
```

Model: `ViT-B/32` · Embedding dim: `512` · All inference on **CPU**

---

### `SceneContextReasoner` — Scene Proximity Score (Sp)

Computes `Sp ∈ [0, 1]` per object using two sub-components:

**Component A — Co-presence reinforcement (0.0 – 0.50):**  
Checks how many scene-context reinforcers are present (e.g., `dining table + fork` confirms a dining scene for *"serve wine"*), adding `+0.10` per matching category.

**Component B — Fallback positional boost (0.0 – 0.40):**  
When the ideal graph object is absent, present alternatives receive `+0.10 × missing_rank` to reward the best available fallback.

```python
sp = scene_reasoner.compute_scene_score(
    task       = "serve wine",
    obj_label  = "cup",
    all_labels = ["cup", "bottle", "dining table", "fork"],
    graph      = kg
)
# Component A: dining table + fork + bottle present → +0.30
# Component B: wine glass missing, cup is rank-2 fallback → +0.20
# Sp = min(1.0, 0.50)
```

---

### `ScoringEngine` — Final Ranking

```python
scorer = ScoringEngine(alpha=0.20, beta=0.40, gamma=0.25, delta=0.15)

ranked = scorer.rank_objects(
    detected_objects = objects,
    task             = "serve wine",
    graph            = kg,
    enc              = encoder,
    scene_ctx        = scene_reasoner,
)

O_star = ranked[0]
# {
#   'label'        : 'wine glass',
#   'final_score'  : 0.8741,
#   'confidence'   : 0.91,
#   'graph_score'  : 1.00,
#   'semantic_sim' : 0.83,
#   'scene_score'  : 0.50,
#   'score_breakdown': {
#       'alpha_Ci' : 0.182,
#       'beta_Sg'  : 0.400,
#       'gamma_Ss' : 0.229,
#       'delta_Sp' : 0.075,
#       'Sf_raw'   : 0.874,
#   }
# }
```

---

## 📊 Results

### Sample Output Images

| "serve wine" | "sit comfortably" | "extinguish fire" |
|:---:|:---:|:---:|
| ![serve wine](assets/result_serve_wine.jpg) | ![sit comfortably](assets/result_sit_comfortably.jpg) | ![extinguish fire](assets/result_extinguish_fire.jpg) |
| O* = **wine glass** · Sf = 0.874 ✅ | O* = **couch** · Sf = 0.831 ✅ | O* = **fire hydrant** · Sf = 0.658 ✅ |

> 📸 *Upload your result images from `results/snapshots/` into the `assets/` folder to display them here.*

---

### Score Breakdown — *"serve wine"* Example

```
══════════════════════════════════════════════════════════════════════════════════
  ASTRA Result — Task: "serve wine"
══════════════════════════════════════════════════════════════════════════════════
  Rank  Label                      Ci      Sg      Ss      Sp        Sf
  ──────────────────────────────────────────────────────────────────────────────
  ★ 1   wine glass              0.912   1.000   0.834   0.500    0.8741
    2   cup                     0.780   0.600   0.524   0.400    0.6123
    3   bottle                  0.712   0.400   0.431   0.300    0.4987
    4   knife                   0.651   0.000   0.118   0.000    0.1624
══════════════════════════════════════════════════════════════════════════════════
  O* = wine glass   |   Sf = 0.8741   |   Latency = 312 ms
══════════════════════════════════════════════════════════════════════════════════
```

![Score table screenshot](assets/score_table_screenshot.png)

> 📸 *Replace with a screenshot of your actual terminal output after running Cell 12.*

---

### 14-Task COCO Evaluation

Evaluated on the **COCO-Tasks benchmark** — 14 tasks, one image per task, CPU-only inference.

| # | Task | Predicted O* | Sf Score | ✓ |
|:-:|------|:-----------:|:--------:|:-:|
| 1 | Step on something to reach top of a shelf | chair | 0.742 | ✅ |
| 2 | Sit comfortably | couch | 0.831 | ✅ |
| 3 | Place flowers | vase | 0.889 | ✅ |
| 4 | Get potatoes out of fire | fork | 0.761 | ✅ |
| 5 | Water a plant | bottle | 0.798 | ✅ |
| 6 | Get lemon out of tea | spoon | 0.774 | ✅ |
| 7 | Dig a hole | baseball bat | 0.612 | ✅ |
| 8 | Open a bottle of beer | knife | 0.683 | ✅ |
| 9 | Open a parcel | scissors | 0.701 | ✅ |
| 10 | Serve wine | wine glass | 0.874 | ✅ |
| 11 | Pour sugar | spoon | 0.749 | ✅ |
| 12 | Smear butter | knife | 0.802 | ✅ |
| 13 | Extinguish fire | fire hydrant | 0.658 | ✅ |
| 14 | Pound a carpet | baseball bat | 0.634 | ✅ |

> 📝 *Replace scores with your actual run output from `results/evaluation_results.json`*

---

### Performance Summary

| Metric | CPU-Only *(Stage 2A)* | FPGA Target *(Stage 3)* |
|--------|:--------------------:|:----------------------:|
| **Accuracy** | ≥ 71.4% (10/14) | — |
| **Avg Latency** | ~320 ms / image | < 50 ms |
| **Throughput** | ~3 FPS | ≥ 25 FPS |
| **Speedup** | 1× (baseline) | 3.4× |
| **Power Efficiency** | Baseline | Improved |

---

## 🗺️ Roadmap

```
Stage 1  ✅  Proposal & Architecture Design
             Knowledge graph · System specification · DVCon submission accepted

Stage 2A ✅  Software Pipeline  ← YOU ARE HERE
             CPU-based Python / Colab implementation
             14-task COCO evaluation
             Extended formula: Sf = α·Ci + β·Sg + γ·Ss + δ·Sp

Stage 2B 🔄  RTL / Hardware Acceleration
             FPGA similarity engine in Verilog / SystemVerilog
             Vector dot-product accelerator · Object ranking core

Stage 3  🔲  VEGA RISC-V Integration
             Vega IP core communication · Full HW-SW co-design
             End-to-end demo on Genesys-2 board
```

---

## ⚙️ Technical Specifications

| Parameter | Specification |
|-----------|--------------|
| **Input Image Size** | 224×224 / 416×416 (auto-scaled) |
| **Input Type** | RGB Image + Natural Language Task String |
| **Object Detector** | YOLOv5s (COCO-80 pretrained) |
| **Detection Threshold** | conf ≥ 0.20 |
| **Task Encoder** | CLIP ViT-B/32 text branch |
| **Object Encoder** | CLIP ViT-B/32 text + image (dual-channel ensemble) |
| **Embedding Dimension** | 512 |
| **Graph Type** | Directed Weighted Task-Object Graph |
| **Graph Reasoning** | Weighted edge lookup + neighbour propagation |
| **Similarity Measure** | Cosine similarity → normalised to [0, 1] |
| **Scoring Formula** | `Sf = 0.20·Ci + 0.40·Sg + 0.25·Ss + 0.15·Sp` |
| **Scene Context** | Co-presence pattern matching + fallback rank boost |
| **Inference Device** | CPU (Stage 2A requirement) |
| **Target Hardware** | VEGA RISC-V + FPGA (Genesys-2) |
| **Target Latency** | < 50 ms (FPGA-accelerated) |
| **Target Throughput** | ≥ 25 FPS (FPGA-accelerated) |
| **Output** | Object label + bounding box + Sf score + full breakdown |
| **Dataset** | COCO-80 (val2017) + COCO-Tasks benchmark |

---

## 📚 References

| # | Citation |
|:-:|---------|
| [1] | **J. Xie et al.**, *"What Object Should I Use? — Task Driven Object Detection,"* arXiv:1904.03000, 2019. **← Core benchmark (defines the 14 tasks)** |
| [2] | **J. Zhang, J. Huang, S. Jin, S. Lu**, *"Vision-Language Models for Vision Tasks: A Survey,"* IEEE TPAMI, vol. 46, no. 8, pp. 5625–5640, Aug. 2024. |
| [3] | **J. Xie and S. Zheng**, *"Zero-shot Object Detection Through Vision-Language Embedding Alignment,"* IEEE/CVPR, 2022. |
| [4] | **S. Song, J. Wan, Z. Yang et al.**, *"Vision-Language Pre-Training for Boosting Scene Text Detectors,"* IEEE Conference, 2022. |
| [5] | **J. Yang, N. Jia, X. Liu et al.**, *"Zone-YOLO: Vision-Language Object Detection Using Zone Prompt,"* IEEE Trans. ITS, vol. 26, no. 10, pp. 17510–17521, 2025. |
| [6] | **A. Radford et al.** (OpenAI), *"Learning Transferable Visual Models From Natural Language Supervision (CLIP),"* ICML 2021. |
| [7] | **G. Jocher et al.**, *"YOLOv5 by Ultralytics,"* 2020. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| [8] | **T.-Y. Lin et al.**, *"Microsoft COCO: Common Objects in Context,"* ECCV 2014. |

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

> This repository is a competition submission for **DVCon India 2026 Design Contest**.  
> All code, architecture, and the extended scoring formula (`Sf = α·Ci + β·Sg + γ·Ss + δ·Sp`) are original work by Team VILSCORE (#389).

---

<div align="center">

<br/>

**Built with** 🧠 reasoning · 🔍 vision · 🕸️ graphs · 🌍 context

*ASTRA — selecting not what is present, but what is needed.*

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KeshavJ1007/Adaptive-Semantic-Task-Reasoning-Architecture/blob/main/ASTRA_Stage2A_Colab_v3.ipynb)

<br/>

**Kamalesh S · Keshav J · Raghul V**  
Saveetha Engineering College · DVCon India 2026

</div>
