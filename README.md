# 🧩 Data Science Internship — Project 3
## Unsupervised Learning: Customer Segmentation Pipeline | Industrial Training Kit

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PCA](https://img.shields.io/badge/PCA-Dimensionality%20Reduction-6A5ACD?style=for-the-badge&logo=python&logoColor=white)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-228B22?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-blue?style=for-the-badge)

**Powered by [DecodeLabs](https://www.decodelabs.tech) | Greater Lucknow, India**

</div>

---

## 📌 Short Description

> *"A model that groups customers without proving its groupings is just guessing with extra steps — mathematically-unjustified clusters translate directly into wasted marketing spend."*

**Project 3** is the technical mastery phase of the internship: **Unsupervised Learning**. The goal is to build a **distance-based clustering pipeline** on unlabeled retail data with 20+ correlated features, using **PCA** to compress the feature space, **K-Means** to discover natural customer groupings, and the **Elbow Method + Silhouette Score** to mathematically prove the chosen number of clusters — before translating the raw output into actionable business personas.

---

## 🎯 Project Goal

**Use distance-based algorithms to discover hidden mathematical groupings** in unlabeled retail data, then reverse-engineer those groupings into human-readable, marketing-ready **customer personas**. No accuracy metric exists here — success is measured by cluster cohesion, separation, and business interpretability.

---

## 🏗️ Pipeline Architecture

```
Raw High-Dimensional  ──►  Standardization  ──►  PCA Compression  ──►  K-Means Clustering  ──►  Persona
Data (20+ features)        (StandardScaler)      (95% variance          (Elbow + Silhouette      Translation
                                                   retained)              validated k)             (Business Output)
```

---

## 📂 Repository Structure

```
data-science-internship-project-3/
│
├── 📓 Customer_Segmentation_Project_3.ipynb    ← Main notebook (run this)
├── 📊 customer_data.csv                         ← Unlabeled retail/customer dataset
├── 📄 README.md                                 ← You are here
│
├── outputs/
│   ├── elbow_method.png                         ← WCSS vs K curve
│   ├── silhouette_scores.png                    ← Cluster validation scores
│   ├── pca_variance_explained.png                ← Cumulative explained variance (95% rule)
│   ├── cluster_scatter_2d.png                    ← PCA-space cluster visualization
│   └── persona_matrix.png                        ← Final business persona summary
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | Customer / Retail Behavioral Dataset |
| **Type** | Unlabeled (no target variable — pure unsupervised task) |
| **Features** | 20+ (demographic + transactional: Age, Income, Spending Score, Purchase Frequency, etc.) |
| **Target** | None — clusters are discovered, not predicted |

> **The Core Problem:** With 20+ correlated features, raw Euclidean distance breaks down — the **Curse of Dimensionality** makes all points appear equidistant. Clustering must happen in a compressed space, not the raw one.

---

## 🔬 Key Engineering Concepts

### ⚠️ Trap #1 — Unscaled Features Distort Distance

| Feature | Raw Range | Problem |
|---------|-----------|---------|
| **Annual Income** | $0 – $100,000 | Dominates Euclidean distance by magnitude alone |
| **Purchases/Month** | 0 – 10 | Mathematically swallowed — becomes irrelevant to clustering |

```
❌ Never cluster on raw, unscaled features.
✅ Always apply StandardScaler first — z = (x − μ) / σ
```

Euclidean distance treats all axes equally, so magnitude — not importance — decides influence. Standardization gives every feature equal mathematical voting power before clustering begins.

---

### ⚠️ Trap #2 — The Curse of Dimensionality

In high-dimensional space (D > 20), volume grows exponentially relative to distance, so data points become nearly equidistant and standard distance metrics stop being meaningful.

```
❌ WRONG: Run K-Means directly on 20+ raw features
✅ CORRECT: Compress with PCA first, then cluster in reduced space
```

---

### 🔹 PCA — Finding the Angle of Maximum Variance

PCA is an unsupervised linear transformation that finds orthogonal axes (**Principal Components**) capturing the widest spread of the data — like a light casting the "best angle" shadow of a high-dimensional cloud onto a lower-dimensional surface.

**Eigenvalue Equation:**
```
Σv = λv
```

**The 95% Rule:**
```
Σ(EVR_i) ≥ 0.95   for i = 1 to k
```
A 20+ column dataset is mathematically compressed into just **4–5 principal components** while retaining 95% of the original variance — discarding low-variance noise, keeping core behavioral signal.

---

### 🔹 K-Means — The Iterative Mechanics

| Step | Action |
|------|--------|
| **1. Initialize** | Randomly place K centroids |
| **2. Assign** | Assign every point to its nearest centroid |
| **3. Update** | Recompute each centroid as the mean of its assigned points |
| Repeat 2–3 | Until centroids stabilize |

**Objective — minimize Within-Cluster Sum of Squares (WCSS):**
```
WCSS = Σ ||x - μ_k||²
```

> K-Means cannot determine how many clusters *should* exist — it will force data into whatever K it's given. The number of clusters must be **proven**, not guessed.

---

### 🔹 The "K" Dilemma — Two Diagnostic Gatekeepers

| Method | Metric | Interpretation |
|--------|--------|-----------------|
| **Elbow Method** | WCSS vs K | The "elbow" (point of maximum curvature) marks diminishing returns — where adding more clusters stops adding real behavioral insight |
| **Silhouette Score** | `s(i) = (b(i) - a(i)) / max(a(i), b(i))` | Score near **+1** = excellent cohesion & separation; score near **0** = overlapping, poorly separated clusters |

```
❌ Never pick K by eye or by convenience.
✅ Always validate K with BOTH the Elbow Method AND Silhouette Score.
```

---

### 🔹 Reverse-Engineering the Centroids

K-Means runs inside PCA space — its centroid coordinates are abstract and meaningless to a marketing team. They must be mapped back through the **inverse transforms** of PCA and StandardScaler to reconstruct interpretable metrics (Age, Income, Spending Score).

**Inverse Scaling Projection:**
```
C_original = (C_scaled ⊙ σ) + μ
```

```python
# Reverse-engineer centroids back to original feature scale
centroids_pca_space = kmeans.cluster_centers_
centroids_original_space = scaler.inverse_transform(
    pca.inverse_transform(centroids_pca_space)
)
```

---

### 🔹 Translating Clusters into Business Personas

| Cluster | Persona | Profile | Recommended Action |
|---------|---------|---------|---------------------|
| **0** | The Affluent Conservatives | Age ~41, Income ~$88.5k, Spending Score ~17.2 | High-touch support, warranties, loyalty programs |
| **1** | The High-Value Trendsetters | Age ~33, Income ~$86.5k, Spending Score ~82.1 | Exclusive perks, early access, experiential marketing |
| **2** | The Budget-Conscious Explorers | Age ~25, Income ~$25.7k, Spending Score ~79.4 | Influencer campaigns, flash sales, buy-now-pay-later |
| **3** | The Conservative Minimizers | Age ~45, Income ~$26.3k, Spending Score ~20.9 | Minimize spend, clear price-value, basic utility |

---

## ✅ The Zero-Guesswork Protocol (Complete Checklist)

```
☑  ALWAYS scale features with StandardScaler before computing distance
☑  ALWAYS compress high-dimensional data with PCA before clustering
☑  Retain components up to the 95% cumulative explained variance threshold
☑  NEVER pick K arbitrarily — validate with Elbow Method AND Silhouette Score
☑  Reverse-transform centroids back to original units before interpreting
☑  Translate every cluster into a concrete, actionable business persona
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/NaseerAhmed112/customer-segmentation.git
cd customer-segmentation

# 2. Place your retail/customer dataset in the project root
# (customer_data.csv or equivalent unlabeled dataset)

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch Jupyter
jupyter notebook Customer_Segmentation_Project_3.ipynb

# 5. Run All → Kernel → Restart & Run All
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Pandas` | 2.0+ | Data loading and manipulation |
| `NumPy` | 1.24+ | Numerical operations |
| `Scikit-learn` | 1.3+ | StandardScaler, PCA, KMeans, silhouette_score |
| `Matplotlib` | 3.7+ | Elbow curve, PCA variance plot, cluster scatter plots |
| `Seaborn` | 0.12+ | Feature distributions, persona visualizations |

---

## 📊 Results Summary

```
═══════════════════════════════════════════════════════════════
  PROJECT 3: CUSTOMER SEGMENTATION PIPELINE — FINAL RESULTS
═══════════════════════════════════════════════════════════════
  Dataset          : Unlabeled Retail/Customer Data (20+ features)
  Dimensionality   : Reduced to 4–5 Principal Components (95% variance)
  Clustering       : K-Means, K validated via Elbow Method + Silhouette
  Output           : 4 mathematically-distinct customer clusters
─────────────────────────────────────────────────────────────
  Optimal K            →  4 (confirmed by Elbow + Silhouette)
  Silhouette Score      →  ~0.55+ (well-separated clusters)
  Variance Retained     →  95%+ across selected components
─────────────────────────────────────────────────────────────
  Primary Metric   : Silhouette Score (cluster quality) ✅
  Accuracy         : N/A — unsupervised, no ground-truth labels
═══════════════════════════════════════════════════════════════
```

---

## 📚 Key Skills Demonstrated

```
✅ Unlabeled/Unsupervised Data Handling (no target variable)
✅ Feature Standardization (StandardScaler)
✅ Principal Component Analysis (PCA) for dimensionality reduction
✅ K-Means Clustering (distance-based, iterative optimization)
✅ Elbow Method for optimal K selection
✅ Silhouette Score for cluster validation
✅ Centroid Reverse-Engineering (inverse PCA/scaling transforms)
✅ Business Intelligence Translation (clusters → personas)
✅ 2D/3D PCA Cluster Visualization
✅ Data-Driven Marketing Strategy Recommendations
```

---

## 🏢 About

| | |
|--|--|
| **Program** | Data Science Industrial Training Kit |
| **Project** | Project 3 — Technical Mastery Phase |
| **Track** | Unsupervised Learning & Customer Segmentation |
| **Organization** | DecodeLabs |
| **Batch** | 2026 |
| **Location** | Greater Lucknow, India |
| **Contact** | decodelabs.tech@gmail.com |
| **Website** | [www.decodelabs.tech](https://www.decodelabs.tech) |

---

## 📜 License

This project is part of the DecodeLabs Industrial Training Program and is submitted for educational and certification purposes.

---

<div align="center">

**Made with ❤️ | DecodeLabs Batch 2026**

*"Your journey to becoming a professional Data Scientist reaches a definitive peak right here, right now, with the very first customer persona you extract today."*

</div>
