import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------- User settings ----------
CSV_PATH = r"/content/drive/MyDrive/Classroom/AI class/Advertising.csv"
OUTPUT_CSV = "output_file.csv"
N_CLUSTERS = 3
RANDOM_STATE = 42
# ----------------------------------

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def run_kmeans(df: pd.DataFrame, features, n_clusters=3, random_state=42):
    X = df[features].astype(float).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    df2 = df.copy()
    df2["Cluster"] = kmeans.labels_
    return kmeans, X, df2

def plot_clusters(X, labels, centroids, feature_names=("TV", "Radio")):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, edgecolor="k", alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, linewidths=2,
                edgecolor="k", label="Centroids")
    for i, c in enumerate(centroids):
        plt.annotate(f"C{i}", (c[0], c[1]), textcoords="offset points", xytext=(5,5))
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"K-means clustering (k={len(np.unique(labels))})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()

def main():

    print("Loading dataset...")
    data = load_data(CSV_PATH)

    features = ["TV", "Radio"]
    ensure_columns(data, features)
    print(f"Using features: {features}")

    print(f"Running KMeans with k={N_CLUSTERS} ...")
    kmeans, X, data_with_clusters = run_kmeans(data, features, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)

    centroids = kmeans.cluster_centers_
    print("Cluster Centroids:\n", centroids)

    print("Plotting clusters...")
    plot_clusters(X, data_with_clusters["Cluster"].values, centroids, feature_names=features)

    data_with_clusters.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved clustered data to: {OUTPUT_CSV}")

    print("\nCluster counts:")
    print(data_with_clusters["Cluster"].value_counts().sort_index())

if __name__ == "__main__":
    main()
