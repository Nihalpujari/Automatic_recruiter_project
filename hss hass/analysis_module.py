import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# =========================================================
# Preprocessing
# =========================================================

def prep_features(df, feature_cols, categorical_cols=None):
    df2 = df.copy()

    # One-hot categorical cols
    if categorical_cols:
        df2 = pd.get_dummies(df2, columns=categorical_cols, dummy_na=False)
        new_features = []
        for orig in categorical_cols:
            new_features.extend([c for c in df2.columns if c.startswith(f"{orig}_")])
        feature_cols = [c for c in feature_cols if c not in categorical_cols] + new_features

    # Ensure numeric
    for c in feature_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)

    # Transform + scale
    pt = PowerTransformer(method="yeo-johnson")
    X_pt = pt.fit_transform(df2[feature_cols].values)
    X_scaled = StandardScaler().fit_transform(X_pt)

    return df2, X_scaled, feature_cols


# =========================================================
# Stage 1: KMeans(2) Tiering (neutral naming)
# =========================================================

def stage1_tiers(X_scaled, seed=42):
    km2 = KMeans(n_clusters=2, random_state=seed, n_init=10)
    tier_labels = km2.fit_predict(X_scaled)
    tier_names = {0: "Tier 1", 1: "Tier 2"}
    return tier_labels, tier_names


# =========================================================
# Stage 2: KMeans(3) Profiles + RF confidence + LR fit
# + Job role mapping from centroid strength (data-driven)
# + Model-driven selection types (KMeans on rf_confidence & lr_fit)
# + Model-driven ranking (LinearRegression rep_score)
# =========================================================

def stage2_profiles_model_driven(df_encoded, X_scaled, name_col, pick_tier_label, seed=42):
    mask = (df_encoded["tier_label"] == pick_tier_label)
    if mask.sum() < 12:
        return []

    X_t = X_scaled[mask]
    df_t = df_encoded.loc[mask].copy()

    # --- KMeans(3) profiles ---
    km3 = KMeans(n_clusters=3, random_state=seed, n_init=10)
    prof_labels = km3.fit_predict(X_t)
    centers = km3.cluster_centers_

    df_t["profile_id"] = prof_labels

    # Neutral profile names
    profile_names = {0: "Profile A", 1: "Profile B", 2: "Profile C"}

    # ✅ Data-driven job role mapping (no thresholds)
    # Higher centroid norm => “higher intensity” cluster
    centroid_strength = np.linalg.norm(centers, axis=1)
    order = np.argsort(centroid_strength)  # low -> mid -> high
    job_role_map = {
        int(order[0]): "Developer",
        int(order[1]): "Senior Developer",
        int(order[2]): "Solution Architect"
    }
    df_t["job_role"] = df_t["profile_id"].map(job_role_map)

    # --- RF: membership confidence (predict profile_id) ---
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced_subsample",
        min_samples_leaf=2
    )
    rf.fit(X_t, prof_labels)
    prob = rf.predict_proba(X_t)  # [n, 3]

    assigned_conf = prob[np.arange(len(prof_labels)), prof_labels]
    df_t["rf_confidence"] = assigned_conf

    # --- LR: smooth fit score from distance-to-centroid ---
    dists = np.linalg.norm(X_t - centers[prof_labels], axis=1)
    y = -dists  # higher = closer to centroid

    lr_fit_model = LinearRegression()
    lr_fit_model.fit(X_t, y)
    lr_pred = lr_fit_model.predict(X_t)

    lr_norm = (lr_pred - lr_pred.min()) / (lr_pred.max() - lr_pred.min() + 1e-9)
    df_t["lr_fit"] = lr_norm

    results = []

    # --- per profile: model-driven selection types + model-driven ranking ---
    for pid, pname in profile_names.items():
        sub = df_t[df_t["profile_id"] == pid].copy()
        if sub.empty:
            continue

        # Model-driven selection types using KMeans on 2 signals
        type_features = sub[["rf_confidence", "lr_fit"]].to_numpy()
        if len(sub) >= 6:
            km_type = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(type_features)
            sub["selection_type_id"] = km_type.labels_
            type_name_map = {0: "Selection Type 1", 1: "Selection Type 2", 2: "Selection Type 3"}
        else:
            sub["selection_type_id"] = 0
            type_name_map = {0: "Selection Type 1"}

        # Model-driven ranking: regression “rep_score” from the 2 signals
        rep_X = sub[["rf_confidence", "lr_fit"]].to_numpy()
        rep_y = rep_X.mean(axis=1)  # proxy target (no manual weights)
        rep_model = LinearRegression()
        rep_model.fit(rep_X, rep_y)
        sub["rep_score"] = rep_model.predict(rep_X)

        picked = sub.sort_values("rep_score", ascending=False).head(3)

        for _, r in picked.iterrows():
            results.append({
                "name": r[name_col],
                "tier": f"Tier {pick_tier_label + 1}",
                "profile": pname,
                "job_role": r["job_role"],             # ✅ THIS is what your UI should submit
                "score_%": round(float(r["rep_score"]) * 100, 1),
                "rf_conf_%": round(float(r["rf_confidence"]) * 100, 1),
                "lr_fit_%": round(float(r["lr_fit"]) * 100, 1),
                "selection_type": type_name_map[int(r["selection_type_id"])],
                "reason": "Model-driven: clustered into profiles; job_role mapped by centroid strength; ranked by learned rep_score."
            })

    return results


# =========================================================
# Full pipeline runner
# =========================================================

def apply_two_stage_kmeans_rf_lr_model_driven(df, name_col, feature_cols, categorical_cols=None, seed=42):
    if df.empty or len(df) < 12:
        return []

    df_encoded, X_scaled, _ = prep_features(df, feature_cols, categorical_cols)

    tier_labels, tier_names = stage1_tiers(X_scaled, seed=seed)
    df_encoded["tier_label"] = tier_labels
    df_encoded["tier"] = pd.Series(tier_labels).map(tier_names).values

    # Profile the larger tier for stability (neutral rule)
    unique, counts = np.unique(tier_labels, return_counts=True)
    pick_tier = int(unique[np.argmax(counts)])

    return stage2_profiles_model_driven(
        df_encoded=df_encoded,
        X_scaled=X_scaled,
        name_col=name_col,
        pick_tier_label=pick_tier,
        seed=seed
    )


# =========================================================
# Public API for your Flask app
# =========================================================

def get_top_candidates():
    """
    Returns:
      {"GitHub":[{...}], "StackOverflow":[{...}], "Kaggle":[{...}]}
    Each candidate dict includes:
      name, profile, job_role, score_%, reason, etc.
    """
    final_output = {"StackOverflow": [], "Kaggle": [], "GitHub": []}

    # --- GitHub ---
    if os.path.exists("github_candidates_1.csv"):
        df = pd.read_csv("github_candidates_1.csv")
        gh = (
            df.groupby("username")
              .agg({"total_stars": "sum", "total_forks": "sum", "commits_12m": "sum"})
              .reset_index()
        )
        final_output["GitHub"] = apply_two_stage_kmeans_rf_lr_model_driven(
            gh,
            name_col="username",
            feature_cols=["total_stars", "total_forks", "commits_12m"],
            categorical_cols=None
        )

    # --- StackOverflow ---
    if os.path.exists("stackoverflow_200.csv"):
        df = pd.read_csv("stackoverflow_200.csv").fillna(0)
        df["tags"] = df["top_tags"].apply(lambda x: len(str(x).split(",")))
        name_col = "display_name" if "display_name" in df.columns else df.columns[0]

        final_output["StackOverflow"] = apply_two_stage_kmeans_rf_lr_model_driven(
            df,
            name_col=name_col,
            feature_cols=["reputation", "tags", "avg_score_per_answer"],
            categorical_cols=None
        )

    # --- Kaggle ---
    if os.path.exists("kaggle-preprocessed.csv"):
        df = pd.read_csv("kaggle-preprocessed.csv", index_col=0)
        kg = (
            df.groupby("Author_name")
              .agg({
                  "Upvotes": "sum",
                  "Usability": "mean",
                  "No_of_files": "sum",
                  "Medals": lambda x: x.mode()[0] if not x.mode().empty else "None"
              })
              .reset_index()
        )

        final_output["Kaggle"] = apply_two_stage_kmeans_rf_lr_model_driven(
            kg,
            name_col="Author_name",
            feature_cols=["Upvotes", "Usability", "No_of_files", "Medals"],
            categorical_cols=["Medals"]
        )

    return final_output