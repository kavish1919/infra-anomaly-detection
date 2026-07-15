# src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
# IT Infrastructure Log Anomaly Detection — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Transform raw metric readings into a rich feature set that allows
#   unsupervised models to detect anomalies that are invisible at a single
#   point in time (e.g., memory_leak is a gradual trend, not an outlier value).
#
# FEATURE GROUPS (42 total after engineering):
#   7  raw metric values  (already normalised per-server via z-score)
#   28 rolling statistics (mean + std over 1-hr and 4-hr windows, per metric)
#   7  rate-of-change     (first difference vs previous reading, per metric)
#
# DESIGN DECISIONS:
#   - Per-server z-score: normalise within each server's own history, NOT
#     globally, because Database baseline memory != Cache baseline memory.
#     Global normalisation would flag normal DB readings as anomalies.
#   - Rolling windows: 1-hr (4 readings) captures short spikes; 4-hr (16)
#     captures slow drifts like memory_leak. Both are needed.
#   - Rate of change: first-difference catches sudden step changes that the
#     rolling mean might smooth over (e.g. rapid cpu_spike onset).
#   - Label columns are kept in the saved CSV but NEVER passed into any model.
#     An assertion enforces this in Notebook 03.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path

# Columns that are model inputs (raw metrics)
NUMERIC_COLS = [
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "disk_io_mbps",
    "network_latency_ms",
    "response_time_ms",
    "error_rate_pct",
    "requests_per_min",
]

# Columns that are ground-truth labels — kept in the CSV but NEVER model inputs
LABEL_COLS = ["is_anomaly", "anomaly_type"]

# Columns used for identification/grouping only
ID_COLS = ["timestamp", "server_id", "service_type"]

# Rolling window sizes (in number of 15-min readings)
ROLLING_WINDOWS = {
    "1h":  4,   # 4 readings × 15 min = 1 hour
    "4h": 16,   # 16 readings × 15 min = 4 hours
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Rolling statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for each metric over 1-hr and 4-hr windows.

    WHY THIS MATTERS:
      A memory_leak increments memory by only ~0.4% per reading — undetectable
      as a point outlier. But the 4-hr rolling mean will trend steadily upward,
      making the leak visible to clustering algorithms.

    Features added per metric per window: {metric}_roll_mean_{w}, {metric}_roll_std_{w}
    Total new columns: 7 metrics × 2 windows × 2 stats = 28

    Operations are performed per-server (groupby) so window calculations don't
    bleed across server boundaries.
    """
    result_parts = [df.copy()]

    for window_name, window_size in ROLLING_WINDOWS.items():
        for metric in NUMERIC_COLS:
            # Rolling mean — captures trend level
            roll_mean = (
                df.groupby("server_id")[metric]
                .transform(lambda x: x.rolling(window_size, min_periods=window_size).mean())
            )
            # Rolling std — captures volatility; spikes increase local std
            roll_std = (
                df.groupby("server_id")[metric]
                .transform(lambda x: x.rolling(window_size, min_periods=window_size).std())
            )

            df[f"{metric}_roll_mean_{window_name}"] = roll_mean
            df[f"{metric}_roll_std_{window_name}"]  = roll_std

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Rate of change (first difference)
# ─────────────────────────────────────────────────────────────────────────────
def compute_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first-difference features: value at t minus value at t-1, per server.

    WHY THIS MATTERS:
      A cpu_spike jumps from ~30% to ~75% in a single reading. The raw value
      may not be extreme relative to other servers' peaks, but the delta (+45%)
      is far outside normal reading-to-reading variation (~±2–3%).

    Features added: {metric}_diff  (7 columns)
    First row per server will be NaN — these rows are dropped in build_feature_table().
    """
    for metric in NUMERIC_COLS:
        df[f"{metric}_diff"] = df.groupby("server_id")[metric].diff()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Per-server z-score normalisation
# ─────────────────────────────────────────────────────────────────────────────
def normalize_per_server(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Z-score normalise every feature column within each server's own history.

    FORMULA: z = (x - server_mean) / server_std

    WHY PER-SERVER (not global):
      Database servers run at 70% memory baseline; Cache servers at 77%.
      Global normalisation would treat a DB server at 68% as near-normal,
      but a Cache server at 68% as anomalously low. Per-server z-score ensures
      each reading is compared against *that server's own* typical behaviour.

    Columns with zero variance (constant metric) are filled with 0.0 to avoid
    division-by-zero errors.

    NOTE: Applied column-by-column for pandas >= 2.0 compatibility.
    """
    for col in feature_cols:
        def _zscore_col(series):
            mu  = series.mean()
            std = series.std()
            if std == 0 or np.isnan(std):
                return series * 0.0  # constant column -> all zeros
            return (series - mu) / std

        df[col] = df.groupby("server_id")[col].transform(_zscore_col)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_table(
    raw_df: pd.DataFrame,
    output_path: str = "../data/processed/features.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Compute rolling mean/std (1-hr and 4-hr windows per metric)
    2. Compute first-difference (rate of change per metric)
    3. Drop rows with NaN (first ~16 rows per server from rolling windows)
    4. Z-score normalise all feature columns per server
    5. Separate feature matrix (X) from label columns (y)
    6. Save features.csv (X + y concatenated, labels clearly named)

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of generate_full_dataset() — includes is_anomaly and anomaly_type.
    output_path : str
        Where to save features.csv.

    Returns
    -------
    feature_df : pd.DataFrame
        Normalised feature matrix (42 columns). NO label columns.
    labels_df : pd.DataFrame
        Ground-truth labels + identifiers. Kept separate from feature_df.

    ⚠️  LABEL ISOLATION: label columns are NEVER inside feature_df.
        The calling notebook must assert this before passing to any model.
    """
    df = raw_df.copy()

    print(f"Input shape: {df.shape}")

    # Step 1: Rolling features
    print("Step 1: Computing rolling statistics (1-hr, 4-hr)...")
    df = compute_rolling_features(df)

    # Step 2: Rate of change
    print("Step 2: Computing rate-of-change (first difference)...")
    df = compute_rate_of_change(df)

    # Step 3: Drop NaN rows (first window_size rows per server have no rolling values)
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"Step 3: Dropped {n_dropped} rows with NaN (first {n_dropped // 15} per server). "
          f"Remaining: {len(df):,}")

    # Identify all engineered feature columns
    feature_cols = (
        NUMERIC_COLS
        + [f"{m}_roll_mean_{w}" for w in ROLLING_WINDOWS for m in NUMERIC_COLS]
        + [f"{m}_roll_std_{w}"  for w in ROLLING_WINDOWS for m in NUMERIC_COLS]
        + [f"{m}_diff"          for m in NUMERIC_COLS]
    )

    print(f"Step 4: Normalising {len(feature_cols)} features per server...")
    df = normalize_per_server(df, feature_cols)

    # Step 5: Separate features from labels
    labels_df  = df[ID_COLS + LABEL_COLS].copy()
    feature_df = df[feature_cols].copy()

    # Step 6: Save — features + labels concatenated (labels as trailing columns)
    save_df = pd.concat([df[ID_COLS], feature_df, labels_df[LABEL_COLS]], axis=1)
    output  = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(output, index=False)

    print(f"\n[SAVED] features.csv -> {output.resolve()}")
    print(f"Feature matrix shape : {feature_df.shape}")
    print(f"Feature columns      : {len(feature_cols)}")
    print(f"  - Raw metrics      : {len(NUMERIC_COLS)}")
    print(f"  - Rolling features : {len(NUMERIC_COLS) * len(ROLLING_WINDOWS) * 2}")
    print(f"  - Diff features    : {len(NUMERIC_COLS)}")
    print(f"Label columns kept   : {LABEL_COLS}  (NOT in feature matrix)")

    return feature_df, labels_df
