# src/detection_models.py
# ─────────────────────────────────────────────────────────────────────────────
# IT Infrastructure Log Anomaly Detection — Detection Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Provides three unsupervised anomaly detectors:
#
#   KMeansAnomalyDetector   — clusters PCA-reduced features; flags points far
#                             from their nearest centroid (top-X% by distance)
#   DBSCANAnomalyDetector   — density-based; noise points (label=-1) are
#                             anomaly candidates
#   IsolationForestDetector — tree ensemble; works on raw features (no PCA
#                             needed); implemented in Notebook 05
#
# DESIGN NOTES:
#   - K-Means and DBSCAN receive PCA-reduced features because distance metrics
#     degrade in high-dimensional spaces (curse of dimensionality).
#   - Isolation Forest handles high dimensions natively via random partitioning,
#     so PCA pre-processing is not applied to it.
#   - evaluate_detector() is the ONLY function that reads is_anomaly / anomaly_type.
#     It must only be called from Notebook 06.
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class KMeansAnomalyDetector:
    """
    Anomaly detection via K-Means clustering on PCA-reduced features.

    MECHANISM:
      1. Fit K-Means to find 'normal operating' clusters.
      2. For each point, compute its distance to the nearest cluster centroid.
      3. Points with the largest distances are the most anomalous — they don't
         fit neatly into any normal operating pattern.

    THRESHOLD:
      The top `threshold_percentile` percent of distances on the training set
      are flagged as anomalies. This is a tunable hyperparameter — it is NOT
      tuned using ground-truth labels (doing so would break the unsupervised
      project integrity).
    """

    def __init__(self, k: int = 6, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.threshold_ = None

    def fit(self, X_pca: np.ndarray, threshold_percentile: float = 95.0):
        """Fit K-Means and compute the anomaly distance threshold."""
        self.model.fit(X_pca)
        distances = self._centroid_distances(X_pca)
        self.threshold_ = np.percentile(distances, threshold_percentile)
        return self

    def _centroid_distances(self, X_pca: np.ndarray) -> np.ndarray:
        """Euclidean distance from each point to its nearest centroid."""
        labels    = self.model.predict(X_pca)
        centroids = self.model.cluster_centers_
        distances = np.linalg.norm(X_pca - centroids[labels], axis=1)
        return distances

    def score(self, X_pca: np.ndarray) -> np.ndarray:
        """Return anomaly score (distance to nearest centroid) for each point."""
        return self._centroid_distances(X_pca)

    def predict(self, X_pca: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1=anomaly, 0=normal."""
        distances = self._centroid_distances(X_pca)
        return (distances > self.threshold_).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class DBSCANAnomalyDetector:
    """
    Anomaly detection via DBSCAN on PCA-reduced features.

    MECHANISM:
      DBSCAN groups points into dense clusters. Points that don't belong to any
      cluster (labelled -1, called 'noise points') are anomaly candidates —
      they exist in sparse regions of the feature space that don't match any
      normal operating pattern.

    HYPERPARAMETERS:
      eps         — maximum distance between two points to be in the same
                    neighbourhood. Selected via a k-distance plot (knee method).
      min_samples — minimum points to form a dense cluster. Typically 4-10.
      Both are tuned without using ground-truth labels.
    """

    def __init__(self):
        self.model  = None
        self.labels_ = None

    def fit_predict(self, X_pca: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """
        Fit DBSCAN and return binary anomaly labels.
        Points with DBSCAN label -1 (noise) are mapped to 1 (anomaly).
        """
        n_jobs = min(4, os.cpu_count() or 1)
        self.model   = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        raw_labels   = self.model.fit_predict(X_pca)
        self.labels_ = raw_labels
        # Convert: -1 (noise) -> 1 (anomaly), anything else -> 0 (normal)
        return (raw_labels == -1).astype(int)

    def score(self, X_pca: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't produce continuous scores natively.
        We approximate a score using distance to the nearest core point.
        Points farther from any core point get a higher anomaly score.

        Implementation: uses sklearn NearestNeighbors with chunked queries
        to avoid building a full N x M distance matrix (OOM risk on large data).
        """
        if self.model is None or not hasattr(self.model, 'core_sample_indices_'):
            raise RuntimeError("Call fit_predict() before score().")
        core_points = X_pca[self.model.core_sample_indices_]
        if len(core_points) == 0:
            return np.ones(len(X_pca))  # no core points -> all anomalous

        # Fit a 1-NN index on the core points only, then query in chunks
        from sklearn.neighbors import NearestNeighbors as _NN
        n_jobs = min(4, os.cpu_count() or 1)
        nn = _NN(n_neighbors=1, algorithm='auto', n_jobs=n_jobs)
        nn.fit(core_points)

        CHUNK = 5_000  # process 5k rows at a time to cap peak RAM
        dists = np.empty(len(X_pca), dtype=np.float32)
        for start in range(0, len(X_pca), CHUNK):
            end = min(start + CHUNK, len(X_pca))
            d, _ = nn.kneighbors(X_pca[start:end])
            dists[start:end] = d[:, 0]
        return dists


# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class IsolationForestDetector:
    """
    Anomaly detection via Isolation Forest on the full (non-PCA) feature matrix.

    DESIGN DIFFERENCE FROM CLUSTERING METHODS:
      Isolation Forest isolates anomalies by randomly partitioning the feature
      space. It handles high-dimensional data natively because it doesn't rely
      on distance metrics — instead, anomalies are isolated in fewer splits.
      This is why we do NOT apply PCA before Isolation Forest, whereas K-Means
      and DBSCAN benefit from PCA's dimensionality reduction.

    contamination : approximate fraction of anomalies. Set close to the known
      injection rate (~5%) but treated as a tunable hyperparameter, NOT as
      cheating with ground-truth labels.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state  = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=min(4, os.cpu_count() or 1),
        )

    def fit(self, X: np.ndarray):
        """Fit Isolation Forest on the raw feature matrix (no PCA)."""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels: 1=anomaly, 0=normal."""
        raw = self.model.predict(X)        # sklearn: 1=normal, -1=anomaly
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly score (higher = more anomalous).
        sklearn's decision_function returns negative scores for anomalies,
        so we negate it: higher value = more anomalous.
        """
        return -self.model.decision_function(X)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION (only called from Notebook 06)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    detector_name: str,
    anomaly_types: pd.Series = None,
    results_dir: str = "../results",
) -> dict:
    """
    Evaluate a detector against ground-truth labels.

    ⚠️  THIS FUNCTION MUST ONLY BE CALLED FROM NOTEBOOK 06.
        It is the first and only place where is_anomaly labels are used
        for model evaluation.

    Parameters
    ----------
    y_true        : ground-truth binary labels (0/1)
    y_pred        : predicted binary labels (0/1)
    detector_name : string identifier saved to CSV filename
    anomaly_types : Series of anomaly_type strings for per-type breakdown
    results_dir   : where to save CSV files

    Returns
    -------
    dict with precision, recall, f1 (overall) and per_type DataFrame
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    result = {
        "detector":  detector_name,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }

    print(f"\n{'='*50}")
    print(f"  {detector_name}")
    print(f"{'='*50}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Flagged   : {y_pred.sum():,} / {len(y_pred):,} "
          f"({y_pred.mean()*100:.1f}%)")

    # Per-anomaly-type breakdown
    if anomaly_types is not None:
        type_records = []
        for atype in sorted(anomaly_types.unique()):
            if atype == "none":
                continue
            mask   = (anomaly_types == atype).values
            yt_sub = y_true[mask]
            yp_sub = y_pred[mask]
            if yt_sub.sum() == 0:
                continue
            type_records.append({
                "anomaly_type": atype,
                "total_rows":   int(mask.sum()),
                "true_count":   int(yt_sub.sum()),
                "detected":     int(yp_sub.sum()),
                "recall":       round(recall_score(yt_sub, yp_sub, zero_division=0), 4),
                "precision":    round(precision_score(yt_sub, yp_sub, zero_division=0), 4),
                "f1":           round(f1_score(yt_sub, yp_sub, zero_division=0), 4),
            })

        per_type_df = pd.DataFrame(type_records)
        result["per_type"] = per_type_df

        print(f"\n  Per-anomaly-type recall:")
        print(per_type_df[["anomaly_type","true_count","detected","recall","f1"]].to_string(index=False))

        # Save per-type metrics
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        per_type_df.to_csv(f"{results_dir}/{detector_name}_per_type.csv", index=False)

    # Save overall metrics
    overall_df = pd.DataFrame([result]).drop(columns=["per_type"], errors="ignore")
    overall_df.to_csv(f"{results_dir}/{detector_name}_overall.csv", index=False)
    print(f"\n  Saved to results/{detector_name}_overall.csv")

    return result
