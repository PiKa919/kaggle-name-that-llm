"""
Experiment 15: Prompt Group Recovery Audit
Verifies whether unseen test passages answering the same prompt can be recovered
purely from text without prompt_ids, meeting DeepSeek's purity >= 0.95 gate.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV


def audit_group_recovery(threshold: float = 0.35):
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    prompts = df["prompt_id"].values
    texts = df["text"].tolist()
    
    # 1. Feature extraction for prompt recovery: Char 3-5 N-grams + Word TF-IDF
    print("Vectorizing passages for topic clustering...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True
    )
    X = vec.fit_transform(texts)
    
    # 2. Agglomerative Clustering with cosine distance
    print(f"Clustering with distance_threshold={threshold}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(X.toarray())
    n_found = len(np.unique(cluster_labels))
    print(f"Found {n_found} prompt clusters (True count: {len(np.unique(prompts))})")
    
    # 3. Compute Purity: For each cluster, fraction of samples from majority prompt
    purities = []
    cluster_sizes = []
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c)
        c_prompts = prompts[mask]
        cluster_sizes.append(len(c_prompts))
        # Majority prompt share
        majority_count = pd.Series(c_prompts).value_counts().iloc[0]
        purity = majority_count / len(c_prompts)
        purities.append(purity)
        
    mean_purity = np.mean(purities)
    print(f"\nMean Cluster Purity: {mean_purity:.4f} (Gate: >= 0.95)")
    print(f"Cluster Size Stats: Min={min(cluster_sizes)}, Max={max(cluster_sizes)}, Median={np.median(cluster_sizes)}")
    
    # Kolmogorov-Smirnov test on cluster size distribution vs true prompt size distribution
    true_sizes = pd.Series(prompts).value_counts().values
    ks_stat, ks_pval = ks_2samp(cluster_sizes, true_sizes)
    print(f"KS-Test p-value: {ks_pval:.4f} (Gate: > 0.05)")
    
    if mean_purity >= 0.95:
        print("\n>>> PROMPT GROUP RECOVERY GATE: PASSED (Purity >= 0.95) <<<")
    else:
        print(f"\n>>> PROMPT GROUP RECOVERY GATE: FAILED (Purity {mean_purity:.4f} < 0.95) <<<")
        
    return mean_purity, ks_pval


if __name__ == "__main__":
    for th in [0.30, 0.35, 0.40, 0.45]:
        print(f"\n--- Testing Distance Threshold = {th} ---")
        audit_group_recovery(threshold=th)
