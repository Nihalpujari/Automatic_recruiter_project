import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# Core utilities
# =========================

def preprocess(df, feature_cols, categorical_cols=None):
    df2 = df.copy()

    # One-hot encode categoricals (if any)
    final_features = list(feature_cols)
    if categorical_cols:
        df2 = pd.get_dummies(df2, columns=categorical_cols, dummy_na=False)
        expanded = []
        for orig in categorical_cols:
            expanded += [c for c in df2.columns if c.startswith(f"{orig}_")]
        final_features = [c for c in final_features if c not in categorical_cols] + expanded

    # Numeric safety
    for c in final_features:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)

    # Transform + scale
    pt = PowerTransformer(method="yeo-johnson")
    X_pt = pt.fit_transform(df2[final_features].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pt)

    return df2, X_scaled, final_features


def print_top_table(df2, name_col, clusters, rf_conf, lr_fit, n=15):
    tmp = pd.DataFrame({
        "name": df2[name_col].astype(str).values,
        "kmeans_cluster": clusters.astype(int),
        "rf_confidence": rf_conf,
        "lr_fit": lr_fit,
    })
    tmp["rf_conf_%"] = (tmp["rf_confidence"] * 100).round(1)
    tmp["lr_fit_%"] = (tmp["lr_fit"] * 100).round(1)

    # Rank is just for printing convenience (not “decision logic”)
    tmp["rank_score"] = 0.5 * tmp["rf_confidence"] + 0.5 * tmp["lr_fit"]
    tmp = tmp.sort_values("rank_score", ascending=False)

    print("\n[TOP CANDIDATES SNAPSHOT] (for readability only)")
    print(tmp[["name", "kmeans_cluster", "rf_conf_%", "lr_fit_%"]].head(n).to_string(index=False))


# =========================
# Plotting (NO SAVING, ONLY SHOW)
# =========================

def plot_pca_scatter(X_scaled, clusters, title):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, s=30, alpha=0.85)
    plt.title(f"{title} — KMeans Clusters (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


def plot_cluster_sizes(clusters, title):
    unique, counts = np.unique(clusters, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar([str(u) for u in unique], counts)
    plt.title(f"{title} — Cluster Sizes")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.show()


def plot_centroid_heatmap(centers_scaled, feature_names, title):
    # Simple imshow “heatmap” without seaborn
    plt.figure(figsize=(10, 5))
    plt.imshow(centers_scaled, aspect="auto")
    plt.colorbar(label="Centroid value (scaled space)")
    plt.title(f"{title} — KMeans Centroids (Scaled Space)")
    plt.yticks(range(centers_scaled.shape[0]), [f"Cluster {i}" for i in range(centers_scaled.shape[0])])
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_assigned_distance_hist(assigned_dist, clusters, title):
    plt.figure(figsize=(10, 5))
    for cid in np.unique(clusters):
        mask = clusters == cid
        plt.hist(assigned_dist[mask], bins=25, alpha=0.6, label=f"Cluster {cid}")
    plt.title(f"{title} — Distance to Assigned Centroid (by cluster)")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def plot_rf_feature_importance(feature_importances, title, top_n=20):
    items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], vals[::-1])
    plt.title(f"{title} — Random Forest Feature Importances (Top {top_n})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_rf_confidence_hist(rf_conf, title):
    plt.figure(figsize=(9, 5))
    plt.hist(rf_conf, bins=30, alpha=0.85)
    plt.title(f"{title} — RF Confidence Histogram (P(assigned cluster))")
    plt.xlabel("RF confidence")
    plt.ylabel("Count")
    plt.show()


def plot_lr_pred_vs_target(lr_target, lr_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(lr_target, lr_pred, s=18, alpha=0.75)
    plt.title(f"{title} — Linear Regression: Predicted vs Target")
    plt.xlabel("Target (−distance to centroid)")
    plt.ylabel("Predicted")
    plt.show()


def plot_lr_residuals(lr_target, lr_pred, title):
    resid = lr_target - lr_pred
    plt.figure(figsize=(9, 5))
    plt.hist(resid, bins=30, alpha=0.85)
    plt.title(f"{title} — Linear Regression Residuals")
    plt.xlabel("Residual (target − pred)")
    plt.ylabel("Count")
    plt.show()


def plot_feature_distributions_by_cluster(df2, feature_cols, clusters, title, max_features=6):
    # Show up to max_features to avoid “100 graphs” madness
    cols = feature_cols[:max_features]
    for c in cols:
        plt.figure(figsize=(10, 5))
        for cid in np.unique(clusters):
            mask = clusters == cid
            plt.hist(df2.loc[mask, c].values, bins=25, alpha=0.55, label=f"Cluster {cid}")
        plt.title(f"{title} — Feature Distribution by Cluster: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.legend()
        plt.show()


# =========================
# One platform run: KMeans + RF + LR + PRINT + ALL GRAPHS
# =========================

def audit_platform(platform, df, name_col, feature_cols, categorical_cols=None, k=3, seed=42):
    if df.empty or len(df) < 12:
        print(f"\n[{platform}] SKIP: Not enough data (need >= 12 rows).")
        return

    df2, X_scaled, final_features = preprocess(df, feature_cols, categorical_cols)

    # ----- KMeans -----
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters = km.fit_predict(X_scaled)
    centers = km.cluster_centers_
    dists_to_centers = km.transform(X_scaled)
    assigned_dist = dists_to_centers[np.arange(len(clusters)), clusters]

    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    print(f"\n==============================")
    print(f"PLATFORM: {platform}")
    print(f"==============================")
    print("\n[KMEANS OUTPUTS]")
    print(f"n_clusters: {k}")
    print(f"inertia: {km.inertia_:.4f}")
    print(f"iterations: {km.n_iter_}")
    print(f"cluster_sizes: {cluster_sizes}")
    print("centers_scaled shape:", centers.shape)

    # ----- Random Forest (predicts KMeans clusters) -----
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced_subsample",
        min_samples_leaf=2
    )
    rf.fit(X_scaled, clusters)
    rf_proba = rf.predict_proba(X_scaled)
    rf_pred = rf.predict(X_scaled)
    rf_conf = rf_proba[np.arange(len(clusters)), clusters]  # confidence for assigned cluster
    rf_feat_imp = {final_features[i]: float(rf.feature_importances_[i]) for i in range(len(final_features))}

    print("\n[RANDOM FOREST OUTPUTS]")
    print("classes:", [int(x) for x in rf.classes_])
    print("sample rf_proba row[0]:", [round(float(x), 4) for x in rf_proba[0]])
    print("top feature importances:")
    top_imp = sorted(rf_feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    for kname, kval in top_imp:
        print(f"  {kname}: {kval:.4f}")

    # ----- Linear Regression (predicts target = -distance to assigned centroid) -----
    lr_target = -assigned_dist
    lr = LinearRegression()
    lr.fit(X_scaled, lr_target)
    lr_pred = lr.predict(X_scaled)

    r2 = r2_score(lr_target, lr_pred)
    mae = mean_absolute_error(lr_target, lr_pred)

    lr_fit = (lr_pred - lr_pred.min()) / (lr_pred.max() - lr_pred.min() + 1e-9)

    print("\n[LINEAR REGRESSION OUTPUTS]")
    print(f"intercept: {lr.intercept_:.6f}")
    print(f"coef shape: {lr.coef_.shape}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print("sample lr_pred row[0]:", float(lr_pred[0]))

    # ----- Per-row outputs (print a compact top table) -----
    print_top_table(df2, name_col, clusters, rf_conf, lr_fit, n=15)

    # ====== GRAPHS (ALL REASONABLE ONES) ======
    plot_pca_scatter(X_scaled, clusters, platform)
    plot_cluster_sizes(clusters, platform)
    plot_centroid_heatmap(centers, final_features, platform)
    plot_assigned_distance_hist(assigned_dist, clusters, platform)
    plot_rf_feature_importance(rf_feat_imp, platform, top_n=min(20, len(final_features)))
    plot_rf_confidence_hist(rf_conf, platform)
    plot_lr_pred_vs_target(lr_target, lr_pred, platform)
    plot_lr_residuals(lr_target, lr_pred, platform)

    # Feature distributions on RAW aggregated df (limited to avoid plot spam)
    plot_feature_distributions_by_cluster(df2, [c for c in feature_cols if c in df2.columns], clusters, platform, max_features=6)


# =========================
# Load your datasets and run
# =========================

def run_all(k=3, seed=42):
    # --- GitHub ---
    if os.path.exists("github_candidates_1.csv"):
        df = pd.read_csv("github_candidates_1.csv")
        gh = (
            df.groupby("username")
              .agg({"total_stars": "sum", "total_forks": "sum", "commits_12m": "sum"})
              .reset_index()
        )
        audit_platform("GitHub", gh, "username",
                       feature_cols=["total_stars", "total_forks", "commits_12m"],
                       categorical_cols=None, k=k, seed=seed)
    else:
        print("[SKIP] github_candidates_1.csv not found.")

    # --- StackOverflow ---
    if os.path.exists("stackoverflow_200.csv"):
        df = pd.read_csv("stackoverflow_200.csv").fillna(0)
        df["tags"] = df["top_tags"].apply(lambda x: len(str(x).split(",")))
        name_col = "display_name" if "display_name" in df.columns else df.columns[0]
        audit_platform("StackOverflow", df, name_col,
                       feature_cols=["reputation", "tags", "avg_score_per_answer"],
                       categorical_cols=None, k=k, seed=seed)
    else:
        print("[SKIP] stackoverflow_200.csv not found.")

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
        audit_platform("Kaggle", kg, "Author_name",
                       feature_cols=["Upvotes", "Usability", "No_of_files", "Medals"],
                       categorical_cols=["Medals"], k=k, seed=seed)
    else:
        print("[SKIP] kaggle-preprocessed.csv not found.")


if __name__ == "__main__":
    # k=3 => 3 clusters; change to 2 if you want only good/bad
    run_all(k=3, seed=42)