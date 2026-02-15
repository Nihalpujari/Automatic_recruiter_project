import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

def percentile_normalize(df, feature_cols):
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = df_norm[col].rank(pct=True)
    return df_norm

def _safe_minmax(arr):
    arr = np.array(arr, dtype=float)
    minv, maxv = arr.min(), arr.max()
    if maxv - minv < 1e-9:
        return np.zeros_like(arr)
    return (arr - minv) / (maxv - minv)

def _kmeans_fit(X, k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return km, labels, km.cluster_centers_, sil

def plot_pca_scatter(X, labels, title):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=30, alpha=0.85)
    plt.title(f"{title} — PCA(2D) Cluster View")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def plot_cluster_sizes(labels, title):
    u, c = np.unique(labels, return_counts=True)
    plt.figure(figsize=(7, 4))
    plt.bar([str(x) for x in u], c)
    plt.title(f"{title} — Cluster Sizes")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.show()

def plot_centroid_heatmap(centers, feature_cols, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(centers, aspect="auto")
    plt.colorbar(label="Centroid value (scaled space)")
    plt.yticks(range(centers.shape[0]), [f"Cluster {i}" for i in range(centers.shape[0])])
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.title(f"{title} — KMeans Centroids (Scaled)")
    plt.tight_layout()
    plt.show()

def plot_distance_hist(dists, labels, title):
    plt.figure(figsize=(9, 5))
    for cid in np.unique(labels):
        mask = labels == cid
        plt.hist(dists[mask], bins=25, alpha=0.6, label=f"Cluster {cid}")
    plt.title(f"{title} — Distance to Assigned Centroid")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def plot_rf_feature_importance(feature_cols, importances, title, top_n=20):
    items = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(names, vals)
    plt.title(f"{title} — RF Feature Importance (Top {min(top_n, len(feature_cols))})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_pred_vs_target(y, yhat, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(y, yhat, s=18, alpha=0.75)
    plt.title(f"{title} — Predicted vs Target")
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.show()

def plot_residuals(y, yhat, title):
    resid = y - yhat
    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=30, alpha=0.85)
    plt.title(f"{title} — Residuals Histogram")
    plt.xlabel("Residual (y - yhat)")
    plt.ylabel("Count")
    plt.show()

def plot_score_hist(scores, title):
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=30, alpha=0.85)
    plt.title(f"{title} — Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.show()


def audit_pipeline_one_platform(df, name_col, feature_cols, platform_name="Platform", top_n=10, seed=42):
    if df.empty or len(df) < 15:
        print(f"[SKIP] {platform_name}: not enough data (<15).")
        return

    df0 = df[[name_col] + feature_cols].copy()

    df0 = percentile_normalize(df0, feature_cols)

    X = df0[feature_cols].values
    Xs = StandardScaler().fit_transform(X)

    print("\n" + "=" * 70)
    print(f"{platform_name} — AUDIT OUTPUT (KMeans + RF + LR)")
    print("=" * 70)


    sil_scores = []
    tier_runs = []
    tier_models = []

    for s in range(5):
        km2, labels2, centers2, sil2 = _kmeans_fit(Xs, k=2, seed=seed + s)
        sil_scores.append(sil2)
        tier_runs.append(labels2)
        tier_models.append((km2, centers2))

    best_idx = int(np.argmax(sil_scores))
    tier_labels = tier_runs[best_idx]
    km2_best, centers2_best = tier_models[best_idx]
    sil2_best = sil_scores[best_idx]

    u2, c2 = np.unique(tier_labels, return_counts=True)
    tier_sizes = {int(u): int(c) for u, c in zip(u2, c2)}

    print("\n[KMEANS TIERING k=2]")
    print(f"Best silhouette over 5 runs: {sil2_best:.4f}")
    print(f"Inertia: {km2_best.inertia_:.4f} | Iterations: {km2_best.n_iter_}")
    print("Tier cluster sizes:", tier_sizes)

    pick_tier = int(u2[np.argmax(c2)])
    mask_t = tier_labels == pick_tier
    df_t = df0.loc[mask_t].copy()
    X_t = Xs[mask_t]

    print(f"Picked Tier Label: {pick_tier} (largest tier) | rows in tier: {len(df_t)}")

    plot_pca_scatter(Xs, tier_labels, f"{platform_name} — Tiering (k=2)")
    plot_cluster_sizes(tier_labels, f"{platform_name} — Tiering (k=2)")
    plot_centroid_heatmap(centers2_best, feature_cols, f"{platform_name} — Tiering (k=2)")

    if len(df_t) < 12:
        print(f"[STOP] {platform_name}: picked tier too small (<12).")
        return

    
    sil_scores3, prof_runs, centers_runs, prof_models = [], [], [], []

    for s in range(5):
        km3, labels3, centers3, sil3 = _kmeans_fit(X_t, k=3, seed=seed + s)
        sil_scores3.append(sil3)
        prof_runs.append(labels3)
        centers_runs.append(centers3)
        prof_models.append(km3)

    best_idx3 = int(np.argmax(sil_scores3))
    prof_labels = prof_runs[best_idx3]
    centers3_best = centers_runs[best_idx3]
    km3_best = prof_models[best_idx3]
    sil3_best = sil_scores3[best_idx3]

    u3, c3 = np.unique(prof_labels, return_counts=True)
    prof_sizes = {int(u): int(c) for u, c in zip(u3, c3)}

    print("\n[KMEANS PROFILES k=3 (inside picked tier)]")
    print(f"Best silhouette over 5 runs: {sil3_best:.4f}")
    print(f"Inertia: {km3_best.inertia_:.4f} | Iterations: {km3_best.n_iter_}")
    print("Profile cluster sizes:", prof_sizes)

    dists_to_centers = km3_best.transform(X_t)
    assigned_dist = dists_to_centers[np.arange(len(prof_labels)), prof_labels]

    plot_pca_scatter(X_t, prof_labels, f"{platform_name} — Profiles (k=3)")
    plot_cluster_sizes(prof_labels, f"{platform_name} — Profiles (k=3)")
    plot_centroid_heatmap(centers3_best, feature_cols, f"{platform_name} — Profiles (k=3)")
    plot_distance_hist(assigned_dist, prof_labels, f"{platform_name} — Profiles (k=3)")

    df_t["pct_composite"] = df_t[feature_cols].mean(axis=1)

    rf = RandomForestRegressor(n_estimators=400, random_state=seed)
    rf.fit(X_t, df_t["pct_composite"].values)
    rf_pred = rf.predict(X_t)

    print("\n[RANDOM FOREST REGRESSOR]")
    print("Target: pct_composite (= mean of percentile features)")
    print("Feature importances (top):")
    imp_pairs = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for n, v in imp_pairs[:10]:
        print(f"  {n}: {v:.4f}")

    plot_rf_feature_importance(feature_cols, rf.feature_importances_, f"{platform_name} — RF Importances", top_n=20)
    plot_pred_vs_target(df_t["pct_composite"].values, rf_pred, f"{platform_name} — RF (pred vs target)")
    plot_residuals(df_t["pct_composite"].values, rf_pred, f"{platform_name} — RF residuals")

    lr_target = -assigned_dist  
    lr = LinearRegression()
    lr.fit(X_t, lr_target)
    lr_pred = lr.predict(X_t)
    lr_fit = _safe_minmax(lr_pred)

    print("\n[LINEAR REGRESSION]")
    print("Target: -distance_to_assigned_centroid")
    print(f"Intercept: {lr.intercept_:.6f}")
    print(f"R2: {r2_score(lr_target, lr_pred):.4f} | MAE: {mean_absolute_error(lr_target, lr_pred):.4f}")

    plot_pred_vs_target(lr_target, lr_pred, f"{platform_name} — LR (pred vs target)")
    plot_residuals(lr_target, lr_pred, f"{platform_name} — LR residuals")


    Z = np.column_stack([rf_pred, lr_fit])
    target = df_t["pct_composite"].values

    w_model = LinearRegression(positive=True)
    w_model.fit(Z, target)
    w_rf, w_lr = w_model.coef_

    final_score = _safe_minmax(w_model.predict(Z))
    df_t["final_score"] = final_score

    print("\n[AUTO-WEIGHTED FINAL SCORE]")
    print(f"Learned weights: RF={w_rf:.4f}, LR={w_lr:.4f}")
    plot_score_hist(final_score, f"{platform_name} — Final Score")

    df_t["profile_id"] = prof_labels
    strength = df_t.groupby("profile_id")["final_score"].mean().sort_values()
    ordered_profiles = strength.index.tolist()

    role_map = {
        int(ordered_profiles[0]): "Developer",
        int(ordered_profiles[1]): "Senior Developer",
        int(ordered_profiles[2]): "Solution Architect",
    }
    df_t["job_role"] = df_t["profile_id"].map(role_map)

    print("\n[ROLE MAPPING]")
    print("Mean final_score per profile_id:")
    for pid, m in strength.items():
        print(f"  profile {int(pid)} mean_final_score={m:.4f} → {role_map[int(pid)]}")

    print("\n[TOP CANDIDATES PER ROLE] (printed for audit)")
    out_rows = []
    for role in ["Developer", "Senior Developer", "Solution Architect"]:
        sub = df_t[df_t["job_role"] == role].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("final_score", ascending=False).head(top_n)
        for _, r in sub.iterrows():
            out_rows.append({
                "name": str(r[name_col]),
                "role": role,
                "profile_id": int(r["profile_id"]),
                "final_score": float(r["final_score"]),
                "pct_composite": float(r["pct_composite"]),
            })

    if out_rows:
        print(pd.DataFrame(out_rows).to_string(index=False))
    else:
        print("No rows selected (unexpected).")



def audit_all(seed=42, top_n=10):
    # GitHub
    if os.path.exists("github_candidates_1.csv"):
        df = pd.read_csv("github_candidates_1.csv")
        gh = (
            df.groupby("username")
              .agg({"total_stars": "sum", "total_forks": "sum", "commits_12m": "sum"})
              .reset_index()
        )
        audit_pipeline_one_platform(
            gh, name_col="username",
            feature_cols=["total_stars", "total_forks", "commits_12m"],
            platform_name="GitHub", top_n=top_n, seed=seed
        )
    else:
        print("[SKIP] github_candidates_1.csv not found.")

    # StackOverflow
    if os.path.exists("stackoverflow_200.csv"):
        df = pd.read_csv("stackoverflow_200.csv").fillna(0)
        df["tags"] = df["top_tags"].apply(lambda x: len(str(x).split(",")))
        name_col = "display_name" if "display_name" in df.columns else df.columns[0]
        audit_pipeline_one_platform(
            df, name_col=name_col,
            feature_cols=["reputation", "tags", "avg_score_per_answer"],
            platform_name="StackOverflow", top_n=top_n, seed=seed
        )
    else:
        print("[SKIP] stackoverflow_200.csv not found.")

    # Kaggle
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
        audit_pipeline_one_platform(
            kg, name_col="Author_name",
            feature_cols=["Upvotes", "Usability", "No_of_files"],
            platform_name="Kaggle", top_n=top_n, seed=seed
        )
    else:
        print("[SKIP] kaggle-preprocessed.csv not found.")


if __name__ == "__main__":
    audit_all(seed=42, top_n=10)
