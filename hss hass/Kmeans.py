import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# --- SETTINGS ---
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (18, 12)

def process_ml_pipeline(df, features, platform_name, name_col):
    """
    Applies K-Means Clustering and Random Forest Classification 
    to a specific dataset and generates visual audit graphs.
    """
    print(f"\n[INFO] Analyzing {platform_name} Dataset...")
    
    # 1. Clean & Aggregate Data
    # Group by name/username to get unique candidate profiles
    df_grouped = df.groupby(name_col).agg({f: 'mean' if 'ratio' in f or 'Usability' in f else 'sum' for f in features}).reset_index()
    
    # 2. ML Preprocessing
    # Use Log1p to squash outliers (stars/reputation)
    X_log = np.log1p(df_grouped[features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 3. K-MEANS: Elite vs Standard Cluster
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_grouped['cluster'] = kmeans.fit_predict(X_scaled)
    # Identify Good Cluster automatically (highest feature sum)
    good_cluster = np.argmax(kmeans.cluster_centers_.sum(axis=1))
    df_grouped['segment'] = df_grouped['cluster'].apply(lambda x: 'Elite' if x == good_cluster else 'Standard')

    # 4. RANDOM FOREST: Pseudo-Labeling Roles
    # Logic: High Scale -> Architect | High Impact -> Senior | Others -> Developer
    labels = []
    for _, row in X_log.iterrows():
        if row.iloc[1] > X_log.iloc[:, 1].quantile(0.8): labels.append("Solution Architect")
        elif row.iloc[0] > X_log.iloc[:, 0].quantile(0.6): labels.append("Senior Developer")
        else: labels.append("Developer")
    
    df_grouped['predicted_role'] = labels
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, labels)

    # 5. VISUALIZATION
    fig, axes = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # A. Cluster Projection (PCA)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df_grouped['segment'], 
                    palette={'Elite': '#FFD700', 'Standard': '#4B0082'}, s=120, ax=axes[0, 0])
    axes[0, 0].set_title(f"{platform_name}: Talent Segments (PCA Projection)")

    # B. Feature Importance
    sns.barplot(x=rf.feature_importances_, y=features, palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("Random Forest: Feature Weighting")

    # C. Role Distribution
    sns.countplot(x=df_grouped['predicted_role'], palette="rocket", ax=axes[1, 0], order=["Developer", "Senior Developer", "Solution Architect"])
    axes[1, 0].set_title("AI Role Classification Distribution")

    # D. Metric Density
    for f in features:
        sns.kdeplot(X_log[f], shade=True, ax=axes[1, 1], label=f)
    axes[1, 1].set_title("Log-Normalized Feature Distribution")
    axes[1, 1].legend()

    plt.suptitle(f"ML Intelligence Audit: {platform_name} Talent Pool", fontsize=22, fontweight='bold')
    plt.show()

# --- DATASET LOADERS ---

def run_all_analysis():
    # 1. GITHUB
    if os.path.exists('github_candidates_1.csv'):
        gh_df = pd.read_csv('github_candidates_1.csv')
        process_ml_pipeline(gh_df, ['total_stars', 'total_forks', 'commits_12m'], "GitHub", "username")
    else:
        print("[SKIP] github_candidates_1.csv not found.")

    # 2. STACKOVERFLOW
    if os.path.exists('stackoverflow_200.csv'):
        so_df = pd.read_csv('stackoverflow_200.csv').fillna(0)
        # We use reputation, accepted_answer_ratio, and avg_score_per_answer
        process_ml_pipeline(so_df, ['reputation', 'accepted_answer_ratio', 'avg_score_per_answer'], "StackOverflow", "display_name")
    else:
        print("[SKIP] stackoverflow_200.csv not found.")

    # 3. KAGGLE
    if os.path.exists('kaggle-preprocessed.csv'):
        kg_df = pd.read_csv('kaggle-preprocessed.csv', index_col=0)
        # Map medals to numeric weights for ML
        kg_df['Medals_Numeric'] = kg_df['Medals'].map({'Gold': 5, 'Silver': 3, 'Bronze': 1}).fillna(0)
        # Upvotes, Usability, Medals
        process_ml_pipeline(kg_df, ['Upvotes', 'Usability', 'Medals_Numeric'], "Kaggle", "Author_name")
    else:
        print("[SKIP] kaggle-preprocessed.csv not found.")

if __name__ == "__main__":
    run_all_analysis()