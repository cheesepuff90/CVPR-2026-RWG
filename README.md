# ROI-wise SM vs RWG Analysis

*A small, reproducible pipeline for aggregating per-ROI Spearman correlations and visualizing the gain from robust (RWG) weighting over a simple equal-weight baseline.*

<p align="center">
  <!-- Optional: replace with your own figure -->
  <img src="docs/figures/overview.svg" width="640"
       alt="Overview: load ROI scores → aggregate per ROI → tables + bar plots for SM vs RWG"/>
</p>

---

## 1 ▪ What & Why

| Step | Traditional approach | Problem | This script does |
|------|----------------------|---------|------------------|
| **Per-ROI scores** | Store raw per-architecture×ROI Spearman scores in an Excel/CSV file | Hard to see *ROI-level* trends, lots of noise | Aggregate across architectures to get **ROI-wise averages** |
| **Comparing methods** | Eyeball two columns (“simple” vs “RWG”) | No easy %-gain summary | Compute **Δ% per ROI** and visualize it |
| **Figures/tables** | Manually copy into LaTeX or plot by hand | Error-prone & not reproducible | Auto-generate a **Jupyter-ready table** and **publication-quality PNG** |

The goal: given per-architecture Spearman correlations for a simple baseline (SM) and a robust RWG variant, produce:

- a **clean ROI-ordered table** (`ROI × [SM, RWG, Diff (%)]`), and  
- a **bar plot** showing SM vs RWG per ROI, with `%∆` labels on top.

This is designed to be drop-in for paper figures and appendices.

---

## 2 ▪ Repository layout

A minimal layout using this script might look like:

```text
.
├── roi_analysis.py              # this script (SM vs RWG table + plots)
├── result.xlsx                  # Excel with per-architecture ROI scores
├── roi_avg_spearman_table.png   # auto-generated table PNG
├── simple_robust_example.ipynb  # optional notebook using dict-based API
└── docs/
    └── figures/
        └── overview.svg         # (optional) overview figure for README
