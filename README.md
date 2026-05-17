# Reviews Analysis — Amazon Product Recommender System

An end-to-end recommendation system built on **568K Amazon product reviews**, implementing and comparing multiple collaborative filtering and latent factor models.

## Models Implemented

| Model | RMSE | Description |
|-------|------|-------------|
| Baseline (μ + b_u + b_i) | 0.9341 | Global mean + user/item bias with shrinkage |
| Item-Based CF | **0.7369** | Item-item cosine similarity with min-common-users constraint |
| User-Based CF | 1.3880 | User-user neighborhood — degrades under extreme sparsity |
| TruncatedSVD (50 factors) | 0.9802 | Latent factor model on residual matrix |

## Key Features

- **Sparsity handling**: 99.997% sparse matrix processed with `scipy.sparse`, reducing memory from ~19B entries to ~188K non-zero
- **Noise filtering**: Pruned users/items with <5 interactions to improve signal-to-noise ratio
- **Min-common-users constraint**: Similarity scores only trusted when ≥10 co-raters exist, preventing spurious correlations
- **Recommendation APIs**: `recommend_similar_products(product_id)` and `recommend_for_user(user_id)`
- **Model persistence**: Trained models, encoders, and similarity matrices serialized for deployment

## Quick Start

```bash
pip install -r requirements.txt
python recommender.py
```

The script will:
1. Load and clean `Reviews.csv`
2. Split into train/test (80/20)
3. Train all 4 models and report RMSE
4. Save artifacts to `models/` and `data/`
5. Print example recommendations

## File Structure

```
.
├── recommender.py      # Main pipeline (load, train, evaluate, recommend)
├── requirements.txt    # Python dependencies
├── models/             # Serialized models and encoders
├── data/               # Sparse matrices and indices
└── Reviews.csv         # Raw dataset (not tracked)
```

## Dataset

Amazon product reviews dataset (~568K records, 256K users, 74K products). Not included in the repository — place `Reviews.csv` in the root directory to run.
