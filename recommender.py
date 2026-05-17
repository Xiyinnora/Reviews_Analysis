#!/usr/bin/env python3
"""
Amazon Product Reviews — Recommender System
=============================================
Methods: Item-Based CF, User-Based CF, TruncatedSVD (latent factor)
Data:    ~568K reviews, 256K users, 74K products (99.997% sparse)
"""

import os
import pickle
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1.  Data loading & cleaning
# ---------------------------------------------------------------------------
DATA_PATH = "Reviews.csv"
DATA_DIR  = "data"
MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_USER_INTERACTIONS = 5   # users with fewer reviews are pruned
MIN_ITEM_INTERACTIONS = 5   # products that appear fewer times are pruned
TEST_SIZE             = 0.2
RANDOM_STATE          = 42

LATENT_FACTORS        = 50  # number of SVD components


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV, keep only essential columns, and filter noise."""
    print("[1/4] Loading data …")
    df = pd.read_csv(path, usecols=["UserId", "ProductId", "Score"])
    print(f"       Raw records : {len(df):,}")
    print(f"       Users       : {df['UserId'].nunique():,}")
    print(f"       Products    : {df['ProductId'].nunique():,}")

    # --- score sanity ---
    assert df["Score"].between(1, 5).all(), "Score out of [1,5]"

    # --- filter users with too few interactions ---
    user_counts = df["UserId"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    df = df[df["UserId"].isin(valid_users)]
    print(f"       After user  filter (>={MIN_USER_INTERACTIONS}): "
          f"{df['UserId'].nunique():,} users, {len(df):,} records")

    # --- filter items with too few interactions ---
    item_counts = df["ProductId"].value_counts()
    valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    df = df[df["ProductId"].isin(valid_items)]
    print(f"       After item  filter (>={MIN_ITEM_INTERACTIONS}): "
          f"{df['ProductId'].nunique():,} items, {len(df):,} records")

    # --- deduplicate (keep last rating per user-item pair) ---
    df = df.drop_duplicates(subset=["UserId", "ProductId"], keep="last")
    print(f"       After dedup : {len(df):,} records")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.  Utility helpers
# ---------------------------------------------------------------------------
def encode_ids(df: pd.DataFrame):
    """Map UserId / ProductId to contiguous integer indices."""
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["u_id"] = user_enc.fit_transform(df["UserId"])
    df["i_id"] = item_enc.fit_transform(df["ProductId"])
    return user_enc, item_enc


def build_sparse_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Utility matrix R (users × items) from triplets."""
    return csr_matrix(
        (df["Score"].values, (df["u_id"].values, df["i_id"].values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def cosine_similarity_sparse(mat: csr_matrix) -> np.ndarray:
    """Cosine similarity matrix for items (columns of mat).

    Returns dense matrix of shape (n_items, n_items).
    Handles zero-norm gracefully.
    """
    norm = np.sqrt(np.array(mat.power(2).sum(axis=0)).squeeze())
    norm[norm == 0] = 1e-10  # avoid division by zero
    # normalise columns
    mat_norm = mat.multiply(1.0 / norm)
    # dense similarity: cosine = (M^T M)
    sim = (mat_norm.T @ mat_norm).toarray()
    np.clip(sim, -1, 1, out=sim)
    return sim


# ---------------------------------------------------------------------------
# 3.  Baseline: global mean + user bias + item bias
# ---------------------------------------------------------------------------
class BaselineModel:
    """
    Baseline predictor:  r̂_ui = μ + b_u + b_i

    Biases are estimated via simple shrinkage:
        b_u = (sum of residuals for u) / (n_u + λ)
        b_i = (sum of residuals for i) / (n_i + λ)
    """

    def __init__(self, lambda_reg: float = 10.0):
        self.mu = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.lambda_reg = lambda_reg

    def fit(self, df: pd.DataFrame):
        print("\n[Baseline] Fitting μ + b_u + b_i …")
        self.mu = df["Score"].mean()
        n_users = df["u_id"].max() + 1   # full encoded range
        n_items = df["i_id"].max() + 1

        # user bias
        residuals = df["Score"] - self.mu
        user_avg = residuals.groupby(df["u_id"]).sum()
        user_cnt = residuals.groupby(df["u_id"]).count()
        b_u_arr = np.zeros(n_users)
        for uid, avg, cnt in zip(user_avg.index, user_avg.values, user_cnt.values):
            b_u_arr[uid] = avg / (cnt + self.lambda_reg)
        self.b_u = b_u_arr

        # item bias
        residuals2 = df["Score"] - self.mu - df["u_id"].map(pd.Series(self.b_u, index=range(n_users)))
        item_avg = residuals2.groupby(df["i_id"]).sum()
        item_cnt = residuals2.groupby(df["i_id"]).count()
        b_i_arr = np.zeros(n_items)
        for iid, avg, cnt in zip(item_avg.index, item_avg.values, item_cnt.values):
            b_i_arr[iid] = avg / (cnt + self.lambda_reg)
        self.b_i = b_i_arr

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.mu + self.b_u[df["u_id"]] + self.b_i[df["i_id"]]
        return np.clip(pred, 1, 5)


# ---------------------------------------------------------------------------
# 4.  Item-Based Collaborative Filtering
# ---------------------------------------------------------------------------
class ItemCF:
    """
    Item-based CF predictor.

    For a user u and item i, predict:
        r̂_ui = (Σ_{j ∈ N_i} s_{ij} · r_uj) / Σ_{j ∈ N_i} |s_{ij}|

    where N_i are items rated by u that are most similar to i.
    """

    def __init__(self, min_common: int = 10, k_neighbors: int = 30):
        """
        Parameters
        ----------
        min_common : int
            Minimum number of co-raters required for a similarity to be
            considered reliable ("min common users" constraint).
        k_neighbors : int
            Number of nearest neighbours to use for prediction.
        """
        self.min_common = min_common
        self.k_neighbors = k_neighbors
        self.sim_: Optional[np.ndarray] = None
        self.R_: Optional[csr_matrix] = None
        self.n_items_: int = 0

    def fit(self, R: csr_matrix, df_train: pd.DataFrame):
        """Compute item-item similarity matrix."""
        print("\n[ItemCF] Computing item-item cosine similarity …")
        t0 = time.time()
        self.R_ = R
        self.n_items_ = R.shape[1]

        # raw cosine
        sim = cosine_similarity_sparse(R)

        # enforce min-common constraint: zero out similarities below threshold
        co_rate = (R.T @ R).toarray()  # co-rating counts
        sim[co_rate < self.min_common] = 0.0

        # zero diagonal so we don't recommend the item itself
        np.fill_diagonal(sim, 0.0)

        self.sim_ = sim
        print(f"       Done in {time.time() - t0:.1f}s  shape={sim.shape}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict for a set of test triplets."""
        preds = []
        u_ids = df_test["u_id"].values
        i_ids = df_test["i_id"].values
        R_csr = self.R_

        for user, item in zip(u_ids, i_ids):
            # items rated by this user
            user_row = R_csr[user].toarray().flatten()
            rated_items = np.where(user_row > 0)[0]
            if len(rated_items) == 0:
                preds.append(3.0)  # fallback
                continue

            # similarities between target item and all items the user rated
            sim_scores = self.sim_[item, rated_items]
            user_ratings = user_row[rated_items]

            # keep top-k neighbours
            if self.k_neighbors < len(sim_scores):
                top_k = np.argsort(sim_scores)[-self.k_neighbors:]
                sim_scores = sim_scores[top_k]
                user_ratings = user_ratings[top_k]

            denom = np.abs(sim_scores).sum()
            if denom == 0:
                preds.append(3.0)
            else:
                pred = np.dot(sim_scores, user_ratings) / denom
                preds.append(pred)

        return np.clip(preds, 1, 5)


# ---------------------------------------------------------------------------
# 5.  User-Based Collaborative Filtering
# ---------------------------------------------------------------------------
class UserCF:
    """
    User-based CF predictor.

        r̂_ui = (Σ_{v ∈ N_u} s_{uv} · r_{vi}) / Σ_{v ∈ N_u} |s_{uv}|

    """

    def __init__(self, min_common: int = 10, k_neighbors: int = 30):
        self.min_common = min_common
        self.k_neighbors = k_neighbors
        self.sim_: Optional[np.ndarray] = None
        self.R_: Optional[csr_matrix] = None

    def fit(self, R: csr_matrix, _df_train=None):
        """Compute user-user cosine similarity matrix."""
        print("\n[UserCF] Computing user-user cosine similarity …")
        t0 = time.time()
        self.R_ = R

        # For user-user we need rows as vectors → cosine on row-normalised R
        sim = cosine_similarity_sparse(R.T)  # same function, transposed view

        # min-common constraint
        co_rate = (R @ R.T).toarray()
        sim[co_rate < self.min_common] = 0.0
        np.fill_diagonal(sim, 0.0)

        self.sim_ = sim
        print(f"       Done in {time.time() - t0:.1f}s  shape={sim.shape}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        preds = []
        u_ids = df_test["u_id"].values
        i_ids = df_test["i_id"].values
        R_csr = self.R_

        for user, item in zip(u_ids, i_ids):
            sim_row = self.sim_[user]
            item_col = R_csr[:, item].toarray().flatten()
            mask = item_col > 0
            candidate_users = np.where(mask)[0]

            if len(candidate_users) == 0:
                preds.append(3.0)
                continue

            sim_scores = sim_row[candidate_users]
            ratings = item_col[candidate_users]

            if self.k_neighbors < len(sim_scores):
                top_k = np.argsort(sim_scores)[-self.k_neighbors:]
                sim_scores = sim_scores[top_k]
                ratings = ratings[top_k]

            denom = np.abs(sim_scores).sum()
            if denom == 0:
                preds.append(3.0)
            else:
                pred = np.dot(sim_scores, ratings) / denom
                preds.append(pred)

        return np.clip(preds, 1, 5)


# ---------------------------------------------------------------------------
# 6.  TruncatedSVD (latent factor model)
# ---------------------------------------------------------------------------
class SVDModel:
    """
    TruncatedSVD  →  R ≈ U · Σ · V^T

    Predictions via reconstruction:  r̂_ui = μ + b_u + b_i + (U Σ^{½})_u · (Σ^{½} V^T)_i

    We fit baseline first, then apply SVD to the residuals to capture
    the signal that biases alone cannot explain.
    """

    def __init__(self, n_factors: int = 50, lambda_reg: float = 10.0):
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.baseline_ = BaselineModel(lambda_reg)
        self.svd_: Optional[TruncatedSVD] = None
        self.U_: Optional[np.ndarray] = None
        self.Vt_: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, R: csr_matrix):
        """Fit baseline + SVD on residual matrix."""
        self.baseline_.fit(df)

        print(f"\n[SVD] Fitting TruncatedSVD with {self.n_factors} factors …")
        t0 = time.time()

        # residual matrix
        pred_baseline = self.baseline_.predict(df)
        residuals = df["Score"].values - pred_baseline
        # build sparse residual matrix
        R_resid = csr_matrix(
            (residuals, (df["u_id"].values, df["i_id"].values)),
            shape=R.shape, dtype=np.float32,
        )

        svd = TruncatedSVD(n_components=self.n_factors, random_state=RANDOM_STATE)
        U_s = svd.fit_transform(R_resid)      # (n_users, k)
        Vt = svd.components_                   # (k, n_items)
        sigma = svd.singular_values_            # (k,)

        # Store user-factors and item-factors for fast dot-product prediction
        sqrt_sigma = np.sqrt(sigma)
        self.U_ = U_s * sqrt_sigma             # (n_users, k)
        self.V_ = (Vt.T * sqrt_sigma).T        # (k, n_items)
        self.svd_ = svd

        print(f"       Done in {time.time() - t0:.1f}s")
        print(f"       Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Predict: baseline + latent factor dot product."""
        base_pred = self.baseline_.predict(df_test)
        latent = np.sum(
            self.U_[df_test["u_id"].values] * self.V_[:, df_test["i_id"].values].T,
            axis=1,
        )
        return np.clip(base_pred + latent, 1, 5)


# ---------------------------------------------------------------------------
# 7.  Evaluation
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: np.ndarray):
    """Print and return RMSE for a fitted model."""
    preds = model.predict(X_test)
    err = rmse(y_test, preds)
    print(f"  {name:20s}  RMSE = {err:.4f}")
    return err


# ---------------------------------------------------------------------------
# 8.  Recommendation functions (for production use)
# ---------------------------------------------------------------------------
def recommend_similar_products(
    product_id: str,
    item_sim_matrix: np.ndarray,
    item_encoder: LabelEncoder,
    item_decoder: LabelEncoder,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Given a product_id, return top-N most similar products.

    Parameters
    ----------
    product_id : str  (original ProductId)
    item_sim_matrix : (n_items, n_items) cosine similarity
    item_encoder : LabelEncoder used during training
    item_decoder : inverse mapping (LabelEncoder)
    top_n : int
    """
    if product_id not in item_encoder.classes_:
        print(f"Product {product_id} not found in training set.")
        return pd.DataFrame()

    idx = item_encoder.transform([product_id])[0]
    sim_scores = item_sim_matrix[idx]
    top_idx = np.argsort(sim_scores)[-top_n - 1 :][::-1]  # include self
    top_idx = top_idx[top_idx != idx][:top_n]  # exclude self

    return pd.DataFrame({
        "ProductId": item_decoder.inverse_transform(top_idx),
        "Similarity": sim_scores[top_idx],
    })


def recommend_for_user(
    user_id: str,
    n_recommendations: int,
    R: csr_matrix,
    item_sim_matrix: np.ndarray,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    item_decoder: LabelEncoder,
) -> pd.DataFrame:
    """
    Generate personalized recommendations for a user via ItemCF aggregation.

    Score each unseen item by weighted average of ratings on similar items.
    """
    if user_id not in user_encoder.classes_:
        return pd.DataFrame()

    uid = user_encoder.transform([user_id])[0]
    user_ratings = R[uid].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    if len(rated_items) == 0:
        return pd.DataFrame()

    # For every item, compute predicted score
    n_items = R.shape[1]
    scores = np.zeros(n_items)
    for i in range(n_items):
        if user_ratings[i] > 0:  # already rated → skip
            scores[i] = -1
            continue
        sim = item_sim_matrix[i, rated_items]
        # zero similarity items → no score
        if sim.sum() == 0:
            continue
        scores[i] = np.dot(sim, user_ratings[rated_items]) / np.abs(sim).sum()

    top_items = np.argsort(scores)[-n_recommendations:][::-1]
    top_scores = scores[top_items]
    mask = top_scores > 0
    return pd.DataFrame({
        "ProductId": item_decoder.inverse_transform(top_items[mask]),
        "PredictedScore": top_scores[mask],
    })


# ---------------------------------------------------------------------------
# 9.  Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Amazon Review Recommender System")
    print("=" * 60)

    # ---- 1. Load & clean ----
    df = load_and_clean(DATA_PATH)
    user_enc, item_enc = encode_ids(df)
    n_users = df["u_id"].nunique()
    n_items = df["i_id"].nunique()
    print(f"       Encoded → {n_users:,} users × {n_items:,} items")

    # ---- 2. Train / test split ----
    print("\n[2/4] Splitting train/test …")
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"       Train: {len(train):,}  Test: {len(test):,}")

    R_train = build_sparse_matrix(train, n_users, n_items)

    y_train = train["Score"].values
    y_test  = test["Score"].values

    # ---- 3. Baseline ----
    print("\n" + "─" * 50)
    print("MODEL EVALUATION")
    print("─" * 50)
    baseline = BaselineModel()
    baseline.fit(train)
    evaluate_model("Baseline", baseline, test, y_test)

    # ---- 4. ItemCF (sample for evaluation due to cost) ----
    itemcf = ItemCF(min_common=10, k_neighbors=30)
    itemcf.fit(R_train, train)
    # evaluate on a random subset for speed
    test_sample = test.sample(min(20000, len(test)), random_state=RANDOM_STATE)
    evaluate_model("Item-based CF", itemcf, test_sample, test_sample["Score"].values)

    # ---- 5. UserCF (sample) ----
    usercf = UserCF(min_common=10, k_neighbors=30)
    usercf.fit(R_train)
    evaluate_model("User-based CF", usercf, test_sample, test_sample["Score"].values)

    # ---- 6. SVD ----
    svd = SVDModel(n_factors=LATENT_FACTORS)
    svd.fit(train, R_train)
    evaluate_model("SVD (latent)", svd, test, y_test)

    # ---- 7. Save artifacts ----
    print("\n" + "─" * 50)
    print("SAVING ARTIFACTS")
    print("─" * 50)
    save_npz(f"{DATA_DIR}/R_train.npz", R_train)
    np.save(f"{DATA_DIR}/train_indices.npy", train[["u_id", "i_id", "Score"]].values)
    np.save(f"{DATA_DIR}/test_indices.npy", test[["u_id", "i_id", "Score"]].values)

    with open(f"{MODEL_DIR}/itemcf.pkl", "wb") as f:
        pickle.dump(itemcf, f)
    with open(f"{MODEL_DIR}/svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump({"user_enc": user_enc, "item_enc": item_enc}, f)

    # save similarity matrix as well (used by recommenders)
    np.save(f"{MODEL_DIR}/item_similarity.npy", itemcf.sim_)

    print("       Models and indices saved to ./models/ & ./data/")

    # ---- 8. Example recommendations ----
    print("\n" + "─" * 50)
    print("EXAMPLE RECOMMENDATIONS")
    print("─" * 50)

    # pick a popular product from training
    top_product = train["ProductId"].value_counts().index[0]
    print(f"\n  Top-5 similar to  {top_product}:")
    sim_products = recommend_similar_products(
        top_product, itemcf.sim_, item_enc, item_enc, top_n=5
    )
    if not sim_products.empty:
        print(f"  {'ProductId':<20s}  {'Similarity':>8s}")
        for _, row in sim_products.iterrows():
            print(f"  {row['ProductId']:<20s}  {row['Similarity']:>8.4f}")

    # pick a frequent user
    top_user = train["UserId"].value_counts().index[0]
    print(f"\n  Top-5 recommendations for user {top_user}:")
    recs = recommend_for_user(
        top_user, 5, R_train, itemcf.sim_,
        user_enc, item_enc, item_enc,
    )
    if not recs.empty:
        print(f"  {'ProductId':<20s}  {'PredictedScore':>14s}")
        for _, row in recs.iterrows():
            print(f"  {row['ProductId']:<20s}  {row['PredictedScore']:>14.4f}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
