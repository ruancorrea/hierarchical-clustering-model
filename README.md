# Hierarchical Cluster Model (HCM)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OR-Tools](https://img.shields.io/badge/Solver-OR--Tools%20v9.8%2B-green.svg)](https://developers.google.com/optimization)

Official standalone research codebase for the **Hierarchical Cluster Model (HCM)** — a data-driven, multi-level spatial clustering and routing framework designed for large-scale **Real-Time Capacitated Vehicle Routing Problems (CVRP)** with physical sorting constraints (*Unit Loads*).

---

## 📌 Overview & Architecture

Modern e-commerce and last-mile distribution networks operate under strict physical warehouse sorting infrastructure (Unit Loads / Sortation Bays) and require real-time vehicle dispatching under tight operational time windows.

The **HCM** framework decomposes the complex global routing problem into a 3-phase hierarchical architecture:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           HCM PIPELINE ARCHITECTURE                           │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  [PHASE 1: Offline Spatial Aggregation (H1)]                                  │
│  Historical Delivery Points ──► Spatial Outlier Filtering (Isolation Forest, e)│
│                             ──► Level-1 Macro-Clustering (K-Means, k1)        │
│                             ──► Proportional Unit Load Capacity Allocation    │
│                             ──► Level-2 Micro-Clustering (Sub-clusters)       │
│                                                                               │
│  [PHASE 2: Offline Regional Calibration (H2)]                                 │
│  Historical Dev Batches     ──► Hyperparameter Grid Search (k1, e-rate)       │
│                             ──► Optimal Spatial Topology Calibration          │
│                                                                               │
│  [PHASE 3: Online Real-Time Route Generation (H3)]                            │
│  Daily Dynamic Deliveries   ──► 2-Level Hierarchical Cluster Assignment       │
│                             ──► Vehicle Capacity Constraint Management (UL)   │
│                             ──► Intra-Cluster Exact Routing (OR-Tools TSP)    │
│                             ──► Dispatch-Ready Vehicle Routes & Deliveries    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Methodological Components
1. **Spatial Outlier Filtering ($e$)**: Isolation Forest pruning of peripheral spatial noise during offline training to prevent centroid distortion in dispersed topologies (e.g., $e = 0.020$ in sparse areas vs. $e = 0.000$ in dense urban cores).
2. **2-Level Spatial Hierarchy ($k_1 \to k_2$)**: Level-1 macro-clusters capture broad regional demand density; Level-2 micro-clusters align with individual vehicle capacities.
3. **Physical Unit Load Allocation**: Allocates warehouse sortation bays proportionally to spatial demand density using a largest remainder strategy.
4. **Sub-second Online Dispatch ($H_3$)**: Assigns incoming deliveries via the pre-trained hierarchy and optimizes final vehicle sequences via intra-cluster TSP using Google OR-Tools.

---

## 📦 Benchmark Datasets & Instances

The experimental evaluation is based on real-world e-commerce last-mile operations from Loggi, spanning **11 distribution centers across 3 metropolitan regions in Brazil**:
- **Belém (PA)**: `pa-0`, `pa-1` (Peripheral and dispersed spatial topologies).
- **Brasília (DF)**: `df-0`, `df-1`, `df-2` (Highway corridors and planned satellite districts).
- **Rio de Janeiro (RJ)**: `rj-0` to `rj-5` (Dense metropolitan coastal core with complex terrain).

### Dataset Structure & Google Drive Download Links

| Dataset Component | Description | Included in Repo? | Google Drive Link |
| :--- | :--- | :---: | :---: |
| **Complete National Instances (`instances.zip`)** | All 11 regions (137,500 test deliveries across 55 operational batches + full training sets). | ⬇️ **External** | [Google Drive: Full Instances (`instances.zip`)](https://drive.google.com/file/d/11m2FjOSAjUII1f-L_gKBsUJy-Kldw-46/view?usp=sharing) |


### How to Install the Complete Dataset:
1. **Download the zip files** from the Google Drive links above.
2. **Extract them into their respective directories**:
```bash
# 1. Extract full instances into instances/
unzip instances.zip -d instances/

# 2. (Optional) Extract precomputed distance matrices into distances/cache/
unzip distance_caches.zip -d distances/cache/
```

> **Note**: If distance cache files are omitted, the framework automatically uses exact spherical Great-Circle (Haversine) distance calculations without requiring external server dependencies.

---

## 🚀 Installation & Setup

### Prerequisites
- Python $\ge$ 3.10
- Standard build tools (`pip`, `venv` or `uv`)

### 1. Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/ruancorrea/hierarchical-clustering-model.git
cd hierarchical-clustering-model

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Quickstart Demo

Run the self-contained demonstration script to train HCM on historical instances and solve an operational batch in real-time:

```bash
python quickstart.py
```

### Example Output:
```text
============================================================
           HCM ROUTING RESULTS SUMMARY
============================================================
Instance Name       : batch_0
Deliveries Served   : 2500
Total Routes Created: 96
Total Distance (km) : 1777.86 km
Avg Packages/Route  : 26.04
============================================================
```

---

## 🧪 Running Benchmarks & Experiments

### 1. Single Execution via `application.py`
Run a specific method (`hcm`, `kmeans`, `sweep`, or `greedy_tsp`) on target instances:

```bash
python application.py \
  --train_instances instances/train/pa-0 \
  --eval_instances instances/batches/pa-0/batch_0.json \
  --method hcm \
  --n_clusters 8 \
  --e_rate 0.0 \
  --n_unit_loads 28 \
  --alpha_criteria 0.94
```

### 2. Multi-Seed Benchmark via `benchmark_runner.py`
Run statistical multi-seed evaluations generating consolidated CSVs, Markdown summaries, and ready-to-use LaTeX tables:

```bash
python benchmark_runner.py \
  --train_instances instances/train/pa-0 \
  --eval_instances instances/batches/pa-0 \
  --methods hcm kmeans sweep \
  --seeds 0 1 2 3 4 \
  --params params/default.json \
  --output solutions/benchmarks
```

---

## 🔬 Benchmark Comparison

Summary of empirical performance on standard large-scale benchmarks (55 operational batches, 137,500 deliveries):

| Model | Total Distance (km) | Total Routes | Avg Packages / Route | Std Packages / Route | Online Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CVRP (Static Upper Bound)** | 112,412 | 4,154 | 32.50 | 0.40 | 11.67 min |
| **HCM (Ours)** | **152,339** | **5,134** | **26.78** | **0.51** | **1.70 min** |
| **KG (K-Means Greedy)** | 156,966 | 5,153 | 26.68 | 0.51 | 1.59 min |
| **QRPS (Sweep)** | 205,122 | 5,077 | 27.08 | 0.46 | 1.56 min |
| **GTSP (Greedy TSP)** | 363,832 | 5,642 | 23.93 | 0.31 | 1,171.97 min |

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

If you use this codebase or the Hierarchical Cluster Model in your research, please cite:

```bibtex
@article{silva2026hcm,
  title   = {A hierarchical clustering approach for the incremental capacitated vehicle routing problem},
  author  = {Silva, Ruan H. C. and Vieira, Tiago F. and Pinheiro, Rian G. S. and Nogueira, Bruno},
  year    = {2026},
  note    = {Submitted for publication}
}
```
