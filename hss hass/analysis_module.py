import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ==========================================================
# Utility Functions
# ==========================================================

def percentile_normalize(df, feature_cols):
    """Convert raw metrics into percentile ranks (0-1 scale)."""
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = df_norm[col].rank(pct=True)
    return df_norm


def _safe_minmax(arr):
    """Safe min-max scaling to 0-1."""
    arr = np.array(arr, dtype=float)
    minv, maxv = arr.min(), arr.max()
    if maxv - minv < 1e-9:
        return np.zeros_like(arr)
    return (arr - minv) / (maxv - minv)


def _kmeans_fit(X, k, seed):
    """Run KMeans and return labels, centers, silhouette."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return labels, km.cluster_centers_, sil


# ==========================================================
# Core Pipeline (KMeans + RF + LR) — No scores returned
# ==========================================================

def run_kmeans_rf_lr_autoweight_no_scores(
    df,
    name_col,
    feature_cols,
    platform_name="Platform",
    top_n=3,
    seed=42
):
    """
    Returns list of dicts:
      {platform, name, role, tier, reason}
    No numeric scores are returned.
    """

    if df.empty or len(df) < 15:
        return []

    # 1) Keep only needed cols
    df0 = df[[name_col] + feature_cols].copy()

    # 2) Percentile normalize (fairer across scales)
    df0 = percentile_normalize(df0, feature_cols)

    # 3) Scale
    X = df0[feature_cols].values
    Xs = StandardScaler().fit_transform(X)

    # 4) Tiering: KMeans(2) choose best of 5 runs by silhouette
    sil_scores = []
    tier_runs = []
    for s in range(5):
        labels_tmp, _, sil_tmp = _kmeans_fit(Xs, k=2, seed=seed + s)
        sil_scores.append(sil_tmp)
        tier_runs.append(labels_tmp)

    best_idx = int(np.argmax(sil_scores))
    tier_labels = tier_runs[best_idx]
    sil2 = sil_scores[best_idx]
    df0["tier"] = tier_labels

    # choose larger tier for stability
    unique, counts = np.unique(tier_labels, return_counts=True)
    pick_tier = int(unique[np.argmax(counts)])

    mask_t = df0["tier"].values == pick_tier
    df_t = df0.loc[mask_t].copy()
    X_t = Xs[mask_t]

    if len(df_t) < 12:
        return []

    # 5) Profiles: KMeans(3) choose best of 5 runs by silhouette
    sil_scores3 = []
    prof_runs = []
    centers_runs = []
    for s in range(5):
        labels_tmp, centers_tmp, sil_tmp = _kmeans_fit(X_t, k=3, seed=seed + s)
        sil_scores3.append(sil_tmp)
        prof_runs.append(labels_tmp)
        centers_runs.append(centers_tmp)

    best_idx3 = int(np.argmax(sil_scores3))
    prof_labels = prof_runs[best_idx3]
    prof_centers = centers_runs[best_idx3]
    sil3 = sil_scores3[best_idx3]

    df_t["profile_id"] = prof_labels

    # 6) Independent target: mean of percentile features
    df_t["pct_composite"] = df_t[feature_cols].mean(axis=1)

    # 7) RF regressor predicts composite (independent signal)
    rf = RandomForestRegressor(n_estimators=400, random_state=seed)
    rf.fit(X_t, df_t["pct_composite"])
    rf_pred = rf.predict(X_t)

    # 8) LR fit score: closer to centroid = better fit
    dists = np.linalg.norm(X_t - prof_centers[prof_labels], axis=1)
    y_fit = -dists
    lr_fit = LinearRegression()
    lr_fit.fit(X_t, y_fit)
    lr_pred = _safe_minmax(lr_fit.predict(X_t))

    # 9) Auto-weight RF + LR (learn weights, no manual 0.6/0.4)
    Z = np.column_stack([rf_pred, lr_pred])
    target = df_t["pct_composite"].values

    w_model = LinearRegression(positive=True)
    w_model.fit(Z, target)
    final_score = _safe_minmax(w_model.predict(Z))

    df_t["final_score"] = final_score

    # 10) Role mapping: cluster mean(final_score) low→high
    cluster_strength = (
        df_t.groupby("profile_id")["final_score"]
        .mean()
        .sort_values()
    )
    ordered_profiles = cluster_strength.index.tolist()

    role_map = {
        int(ordered_profiles[0]): "Developer",
        int(ordered_profiles[1]): "Senior Developer",
        int(ordered_profiles[2]): "Solution Architect",
    }
    df_t["job_role"] = df_t["profile_id"].map(role_map)

    # 11) Pick top_n per role (internally by final_score, but not returned)
    role_order = ["Developer", "Senior Developer", "Solution Architect"]
    results = []

    for role in role_order:
        sub = df_t[df_t["job_role"] == role].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("final_score", ascending=False).head(top_n)

        for _, r in sub.iterrows():
            results.append({
                "platform": platform_name,
                "name": r[name_col],
                "role": role,
                "tier": f"Tier {pick_tier + 1}",
                "reason": (
                    f"Selected by KMeans tiering (sil={sil2:.3f}), "
                    f"KMeans profiling (sil={sil3:.3f}), "
                    f"and model-learned ranking (RF+LR)."
                )
            })

    return results


# ==========================================================
# Flask-facing function
# ==========================================================

def get_top_candidates():
    """
    Returns:
      {"GitHub":[{...}], "StackOverflow":[{...}], "Kaggle":[{...}]}
    Each candidate dict includes:
      platform, name, role, tier, reason
    (No numeric score fields.)
    """

    final_output = {"GitHub": [], "StackOverflow": [], "Kaggle": []}

    # -------- GitHub --------
    if os.path.exists("github_candidates_1.csv"):
        df = pd.read_csv("github_candidates_1.csv")
        gh = (
            df.groupby("username")
              .agg({
                  "total_stars": "sum",
                  "total_forks": "sum",
                  "commits_12m": "sum"
              })
              .reset_index()
        )
        final_output["GitHub"] = run_kmeans_rf_lr_autoweight_no_scores(
            gh,
            name_col="username",
            feature_cols=["total_stars", "total_forks", "commits_12m"],
            platform_name="GitHub"
        )

    # -------- StackOverflow --------
    if os.path.exists("stackoverflow_200.csv"):
        df = pd.read_csv("stackoverflow_200.csv").fillna(0)
        df["tags"] = df["top_tags"].apply(lambda x: len(str(x).split(",")))
        name_col = "display_name" if "display_name" in df.columns else df.columns[0]

        final_output["StackOverflow"] = run_kmeans_rf_lr_autoweight_no_scores(
            df,
            name_col=name_col,
            feature_cols=["reputation", "tags", "avg_score_per_answer"],
            platform_name="StackOverflow"
        )

    # -------- Kaggle --------
    if os.path.exists("kaggle-preprocessed.csv"):
        df = pd.read_csv("kaggle-preprocessed.csv", index_col=0)
        kg = (
            df.groupby("Author_name")
              .agg({
                  "Upvotes": "sum",
                  "Usability": "mean",
                  "No_of_files": "sum"
              })
              .reset_index()
        )

        final_output["Kaggle"] = run_kmeans_rf_lr_autoweight_no_scores(
            kg,
            name_col="Author_name",
            feature_cols=["Upvotes", "Usability", "No_of_files"],
            platform_name="Kaggle"
        )

    return final_output