# Customer Segmentation — Data Science Project 3
---
**Unsupervised Learning | K-Means Clustering | PCA | DecodeLabs Data Science Internship, Batch 2026**
---
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-FF6B6B?style=for-the-badge&logo=python&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-228B22?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-blue?style=for-the-badge)
---
Discover hidden customer segments in unlabeled retail data using distance-based clustering, and translate the results into actionable, marketing-ready personas.

---

## 📌 Overview

This project tackles **customer segmentation** — grouping customers into meaningful clusters purely from their behavioral and demographic data, with no pre-existing labels. The goal isn't just to "group numbers," but to uncover hidden market insights and turn raw mathematical clusters into business personas that a marketing team can act on directly.

## 🎯 Objective

- Apply **Principal Component Analysis (PCA)** to reduce 20+ correlated features down to 2–3 dimensions.
- Use the **Elbow Method** and **Silhouette Score** to mathematically justify the optimal number of K-Means clusters.
- Reverse-engineer cluster centroids back into human-readable metrics.
- Translate each cluster into an actionable **customer persona** with a recommended business strategy.

## 🧠 Pipeline (Input → Process → Output)

| Step | Stage | Description |
|------|-------------------|-------------|
| 1 | **Scale** | Standardize features (`StandardScaler`) so no single feature (e.g. income) dominates distance calculations. |
| 2 | **Compress** | Apply PCA to reduce dimensionality while retaining ~95% of cumulative explained variance. |
| 3 | **Cluster** | Run K-Means in PCA space; validate the optimal `k` with the Elbow Method and Silhouette Score. |
| 4 | **Translate** | Inverse-transform cluster centroids back to original feature scale and map each cluster to a business persona. |

## 🛠️ Key Concepts & Techniques

- **StandardScaler** — mathematical standardization to normalize feature variance before clustering.
- **Principal Component Analysis (PCA)** — orthogonal linear transformation that projects high-dimensional data onto axes of maximum variance.
- **K-Means Clustering** — iterative algorithm that minimizes Within-Cluster Sum of Squares (WCSS) to form cohesive, spherical clusters.
- **Elbow Method** — identifies the inflection point in the WCSS-vs-k curve to select the optimal number of clusters.
- **Silhouette Score** — quantifies cluster cohesion vs. separation (score near +1 = well-separated, near 0 = overlapping).
- **Centroid Reverse-Engineering** — inverse PCA/scaling transform to convert abstract cluster centers back into real-world units (age, income, spending score, etc.).

## 📊 Example Output: Customer Personas

| Cluster | Persona | Profile | Recommended Action |
|---------|---------|---------|---------------------|
| 0 | **The Affluent Conservatives** | Age ~41, Income ~$88.5k, low spending score | High-touch support, warranties, loyalty programs |
| 1 | **The High-Value Trendsetters** | Age ~33, Income ~$86.5k, high spending score | Exclusive perks, early access, experiential marketing |
| 2 | **The Budget-Conscious Explorers** | Age ~25, Income ~$25.7k, high spending score | Influencer campaigns, flash sales, buy-now-pay-later |
| 3 | **The Conservative Minimizers** | Age ~45, Income ~$26.3k, low spending score | Minimize spend, clear price-value, basic utility |

## 🗂️ Repository Structure

```
customer-segmentation/
├── Customer_Segmentation_Project_3.ipynb   # Full analysis: EDA → scaling → PCA → K-Means → personas
└── README.md
```

## ⚙️ Tech Stack

- Python
- pandas, numpy
- scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- matplotlib / seaborn (Elbow plot, PCA scatter plots, cluster visualizations)
- Jupyter Notebook

## 🚀 Getting Started

```bash
git clone https://github.com/NaseerAhmed112/customer-segmentation.git
cd customer-segmentation
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Customer_Segmentation_Project_3.ipynb
```

## 🔑 Key Skills Demonstrated

- Dimensionality reduction (PCA)
- Distance-based clustering (K-Means)
- Cluster validation (Elbow Method, Silhouette Score)
- Business intelligence translation — turning statistical output into actionable strategy
