# src/data_generation.py
# ─────────────────────────────────────────────────────────────────────────────
# IT Infrastructure Log Anomaly Detection — Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Generate a realistic synthetic dataset of infrastructure metrics for 15
#   servers over 30 days at 15-minute granularity (~43,200 rows total).
#
#   Anomalies are injected into ~4-6% of time windows and labelled with
#   'is_anomaly' and 'anomaly_type'. These ground-truth labels are used
#   ONLY for final evaluation in Notebook 06 — NEVER as model inputs.
#
# DESIGN DECISIONS:
#   - Per-server seeded RNG: reproducibility without global random state
#   - Diurnal pattern: sine-curve business-hours multiplier, not flat noise
#   - Per-service-type baselines: realistic heterogeneity across server roles
#   - Random-walk drift: no two servers behave identically even within a type
#   - Anomalies are subtle: relative spikes (baseline + delta), not flat values
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path

# ── Fleet definition ──────────────────────────────────────────────────────────
# 15 servers: 5 Web, 3 DB, 3 API Gateway, 2 Cache, 2 Background Worker
FLEET = [
    ("srv-web-01", "Web Server"),
    ("srv-web-02", "Web Server"),
    ("srv-web-03", "Web Server"),
    ("srv-web-04", "Web Server"),
    ("srv-web-05", "Web Server"),
    ("srv-db-01",  "Database"),
    ("srv-db-02",  "Database"),
    ("srv-db-03",  "Database"),
    ("srv-api-01", "API Gateway"),
    ("srv-api-02", "API Gateway"),
    ("srv-api-03", "API Gateway"),
    ("srv-cache-01",  "Cache"),
    ("srv-cache-02",  "Cache"),
    ("srv-worker-01", "Background Worker"),
    ("srv-worker-02", "Background Worker"),
]

# ── Per-service-type baseline parameter ranges ────────────────────────────────
# Each entry: (cpu_mid, mem_mid, disk_mid, latency_mid, rps_mid, error_mid)
# 'mid' = midpoint of the normal operating range for that service type.
# Actual values = mid ± noise; scaled further by diurnal multiplier.
BASELINE_PARAMS = {
    "Web Server":        {"cpu": 45, "mem": 50, "disk": 30,  "latency": 30,  "rps": 350,  "error": 1.2},
    "Database":          {"cpu": 35, "mem": 70, "disk": 80,  "latency": 10,  "rps": 140,  "error": 0.4},
    "API Gateway":       {"cpu": 40, "mem": 45, "disk": 18,  "latency": 25,  "rps": 600,  "error": 0.9},
    "Cache":             {"cpu": 22, "mem": 77, "disk": 10,  "latency": 5,   "rps": 900,  "error": 0.15},
    "Background Worker": {"cpu": 30, "mem": 40, "disk": 45,  "latency": 45,  "rps": 50,   "error": 0.6},
}

# ── Noise scale per metric (std dev as fraction of baseline mid) ───────────────
NOISE_SCALE = {
    "cpu": 0.08,     # ±8% of mid → mild noise so anomalies aren't trivially obvious
    "mem": 0.06,
    "disk": 0.10,
    "latency": 0.12,
    "rps": 0.15,
    "error": 0.20,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Diurnal multiplier
# ─────────────────────────────────────────────────────────────────────────────
def _diurnal_multiplier(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """
    Returns a per-timestamp multiplier in [0.15, 1.0] that models:
      - Business hours (9–19 on weekdays): peak load following a sine curve
      - Off-hours weekday: reduced load (~0.3)
      - Weekends: minimal load (~0.2)

    A small random perturbation is added so the pattern isn't perfectly smooth.
    """
    multipliers = np.zeros(len(timestamps))
    hours = timestamps.hour
    is_weekend = timestamps.dayofweek >= 5  # 5=Sat, 6=Sun

    for i, (ts, hour, weekend) in enumerate(zip(timestamps, hours, is_weekend)):
        if weekend:
            base = 0.20
        elif 9 <= hour <= 19:
            # Sine curve: peaks at ~14:00, tapers at 9 and 19
            base = 0.55 + 0.40 * np.sin(np.pi * (hour - 9) / 10)
        else:
            base = 0.28

        # Small per-timestamp jitter (not accumulated drift — that's separate)
        multipliers[i] = base + rng.normal(0, 0.03)

    return np.clip(multipliers, 0.10, 1.05)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Generate baseline metrics for one server
# ─────────────────────────────────────────────────────────────────────────────
def generate_baseline_metrics(
    server_id: str,
    service_type: str,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate 'normal' (anomaly-free) metric readings for a single server.

    Steps:
      1. Get service-type baseline midpoints from BASELINE_PARAMS
      2. Compute diurnal multiplier per timestamp
      3. Add per-server random-walk drift (accumulated over time, clipped)
      4. Apply Gaussian noise per metric
      5. Clip values to physically sensible ranges

    Returns a DataFrame with columns:
      timestamp, server_id, service_type,
      cpu_utilization_pct, memory_utilization_pct, disk_io_mbps,
      network_latency_ms, response_time_ms, error_rate_pct,
      requests_per_min, is_anomaly, anomaly_type
    """
    n = len(timestamps)
    bp = BASELINE_PARAMS[service_type]

    # Step 1: Diurnal multiplier
    mult = _diurnal_multiplier(timestamps, rng)

    # Step 2: Random-walk drift (small cumulative wander unique to each server)
    # Each server drifts slightly differently even within the same service type
    drift_cpu    = np.cumsum(rng.normal(0, 0.0015, n)).clip(-0.08, 0.08)
    drift_mem    = np.cumsum(rng.normal(0, 0.0010, n)).clip(-0.06, 0.06)
    drift_disk   = np.cumsum(rng.normal(0, 0.0020, n)).clip(-0.10, 0.10)
    drift_lat    = np.cumsum(rng.normal(0, 0.0020, n)).clip(-0.10, 0.10)
    drift_rps    = np.cumsum(rng.normal(0, 0.0025, n)).clip(-0.12, 0.12)
    drift_error  = np.cumsum(rng.normal(0, 0.0010, n)).clip(-0.05, 0.05)

    # Step 3: Compute each metric = baseline_mid × (diurnal + drift) + noise
    def _metric(mid, noise_frac, drift, clip_lo, clip_hi):
        """Compute one metric column: mid × scaled_multiplier + noise."""
        signal = mid * (mult + drift)
        noise  = rng.normal(0, mid * noise_frac, n)
        return np.clip(signal + noise, clip_lo, clip_hi)

    cpu     = _metric(bp["cpu"],     NOISE_SCALE["cpu"],     drift_cpu,   0,   100)
    mem     = _metric(bp["mem"],     NOISE_SCALE["mem"],     drift_mem,   0,   100)
    disk    = _metric(bp["disk"],    NOISE_SCALE["disk"],    drift_disk,  0,   500)
    latency = _metric(bp["latency"], NOISE_SCALE["latency"], drift_lat,   0.5, 1000)
    rps     = _metric(bp["rps"],     NOISE_SCALE["rps"],     drift_rps,   0,   5000)
    error   = _metric(bp["error"],   NOISE_SCALE["error"],   drift_error, 0,   100)

    # response_time_ms is correlated with latency (latency + processing overhead)
    processing_overhead = rng.uniform(1.5, 3.0, n)
    response_time = np.clip(latency * processing_overhead + rng.normal(0, 5, n), 1, 5000)

    df = pd.DataFrame({
        "timestamp":             timestamps,
        "server_id":             server_id,
        "service_type":          service_type,
        "cpu_utilization_pct":   cpu,
        "memory_utilization_pct": mem,
        "disk_io_mbps":          disk,
        "network_latency_ms":    latency,
        "response_time_ms":      response_time,
        "error_rate_pct":        error,
        "requests_per_min":      rps,
        "is_anomaly":            0,
        "anomaly_type":          "none",
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY INJECTION
# ─────────────────────────────────────────────────────────────────────────────
def inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Inject realistic anomalies into a single-server DataFrame.

    Anomaly types and their design rationale:
      cpu_spike      — Point anomaly: short burst of high CPU.
                       Cross-metric: also nudges response_time up.
                       Subtle: uses relative delta, not flat value.

      memory_leak    — Gradual/contextual: memory climbs slowly over 6–12 hrs.
                       Cross-metric: CPU also increases mildly under memory pressure.
                       Hardest to detect: no single timestamp looks extreme.

      latency_spike  — Correlated: latency AND response_time jump together.
                       Uses multipliers that vary by time-of-day, so absolute
                       values differ, making threshold detection unreliable.

      error_cascade  — Correlated inverse: error_rate spikes while rps drops.
                       Realistic: users failing → fewer completed requests.

      disk_bottleneck — Plateau (not spike): sustained high disk_io for 1–3 hrs.
                       Also moderately raises response_time.

    All anomaly rows are labelled is_anomaly=1 and anomaly_type=<type>.
    Normal rows remain is_anomaly=0, anomaly_type='none'.

    ⚠️  CRITICAL: These ground-truth labels are for final evaluation ONLY.
        They must NOT be used during feature engineering, PCA, or clustering.
    """
    df = df.copy()
    n = len(df)

    # Anomaly types to inject (equal probability)
    anomaly_types = ["cpu_spike", "memory_leak", "latency_spike", "error_cascade", "disk_bottleneck"]

    # ── Pick candidate start indices ──────────────────────────────────────────
    # Average durations per type (readings):
    #   cpu_spike: ~2.5, memory_leak: ~36, latency_spike: ~2, error_cascade: ~4, disk_bottleneck: ~8
    # Blended average: ~10.5 readings per event
    # Target anomalous rows = n * target_anomaly_rate
    # Number of events needed = target_anomalous_rows / avg_duration
    avg_duration = 10.5
    target_anomaly_rows = n * target_anomaly_rate
    num_candidates = max(5, int(target_anomaly_rows / avg_duration))

    # Sample candidate start indices without replacement, spaced out to reduce overlap
    candidate_starts = sorted(rng.choice(n, size=min(num_candidates * 3, n // 2), replace=False))

    injected_mask = np.zeros(n, dtype=bool)  # track which rows are already anomalous
    events_injected = 0  # stop once we hit the target event count

    for start_idx in candidate_starts:
        # Stop if we've already created enough anomaly events
        if events_injected >= num_candidates:
            break

        # Skip if this row is already inside a previous anomaly window
        if injected_mask[start_idx]:
            continue

        atype = rng.choice(anomaly_types)

        # ── CPU Spike: 1–4 readings (15–60 min) ──────────────────────────────
        if atype == "cpu_spike":
            duration = int(rng.integers(1, 5))  # readings
            end_idx  = min(start_idx + duration, n)
            idx      = slice(start_idx, end_idx)

            cpu_baseline = df.loc[start_idx, "cpu_utilization_pct"]
            spike_delta  = rng.uniform(30, 50)  # relative spike, not flat ceiling

            df.loc[start_idx:end_idx - 1, "cpu_utilization_pct"] = np.clip(
                cpu_baseline + spike_delta + rng.normal(0, 3, end_idx - start_idx), 0, 99
            )
            # Mild response_time increase (correlated cross-metric signal)
            df.loc[start_idx:end_idx - 1, "response_time_ms"] *= rng.uniform(1.15, 1.35)

            df.loc[start_idx:end_idx - 1, "is_anomaly"]   = 1
            df.loc[start_idx:end_idx - 1, "anomaly_type"] = "cpu_spike"
            injected_mask[start_idx:end_idx] = True
            events_injected += 1

        # ── Memory Leak: 24–48 readings (6–12 hours) ─────────────────────────
        elif atype == "memory_leak":
            duration = int(rng.integers(24, 49))  # readings
            end_idx  = min(start_idx + duration, n)
            actual_duration = end_idx - start_idx

            mem_start = df.loc[start_idx, "memory_utilization_pct"]
            # Ramp rate: +0.3 to +0.6 % per reading → slow enough to be subtle
            ramp_rate = rng.uniform(0.3, 0.6)
            ramp      = np.arange(actual_duration) * ramp_rate

            new_mem = np.clip(mem_start + ramp + rng.normal(0, 0.5, actual_duration), 0, 96)
            df.loc[start_idx:end_idx - 1, "memory_utilization_pct"] = new_mem

            # Mild CPU increase under memory pressure (cross-metric, subtle)
            cpu_increase = np.linspace(0, rng.uniform(5, 12), actual_duration)
            df.loc[start_idx:end_idx - 1, "cpu_utilization_pct"] = np.clip(
                df.loc[start_idx:end_idx - 1, "cpu_utilization_pct"].values + cpu_increase, 0, 100
            )

            df.loc[start_idx:end_idx - 1, "is_anomaly"]   = 1
            df.loc[start_idx:end_idx - 1, "anomaly_type"] = "memory_leak"
            injected_mask[start_idx:end_idx] = True
            events_injected += 1

        # ── Latency Spike: 1–3 readings (15–45 min) ──────────────────────────
        elif atype == "latency_spike":
            duration = int(rng.integers(1, 4))
            end_idx  = min(start_idx + duration, n)
            actual_duration = end_idx - start_idx

            lat_baseline = df.loc[start_idx, "network_latency_ms"]
            rt_baseline  = df.loc[start_idx, "response_time_ms"]

            # Multipliers vary so absolute values differ by time-of-day
            lat_mult = rng.uniform(3.0, 6.0, actual_duration)
            rt_mult  = rng.uniform(2.5, 5.0, actual_duration)

            df.loc[start_idx:end_idx - 1, "network_latency_ms"] = np.clip(
                lat_baseline * lat_mult, 0, 5000
            )
            df.loc[start_idx:end_idx - 1, "response_time_ms"] = np.clip(
                rt_baseline * rt_mult, 0, 10000
            )
            # Mild error rate increase (weaker signal, realistic)
            df.loc[start_idx:end_idx - 1, "error_rate_pct"] += rng.uniform(1.0, 3.0, actual_duration)
            df.loc[start_idx:end_idx - 1, "error_rate_pct"] = df.loc[
                start_idx:end_idx - 1, "error_rate_pct"
            ].clip(0, 100)

            df.loc[start_idx:end_idx - 1, "is_anomaly"]   = 1
            df.loc[start_idx:end_idx - 1, "anomaly_type"] = "latency_spike"
            injected_mask[start_idx:end_idx] = True
            events_injected += 1

        # ── Error Cascade: 2–6 readings (30–90 min) ──────────────────────────
        elif atype == "error_cascade":
            duration = int(rng.integers(2, 7))
            end_idx  = min(start_idx + duration, n)
            actual_duration = end_idx - start_idx

            error_baseline = df.loc[start_idx, "error_rate_pct"]
            rps_baseline   = df.loc[start_idx, "requests_per_min"]
            rt_baseline    = df.loc[start_idx, "response_time_ms"]

            # Error rate spikes up; requests drop simultaneously (correlated inverse)
            df.loc[start_idx:end_idx - 1, "error_rate_pct"] = np.clip(
                error_baseline + rng.uniform(5, 15, actual_duration), 0, 30
            )
            df.loc[start_idx:end_idx - 1, "requests_per_min"] = np.clip(
                rps_baseline * rng.uniform(0.2, 0.5, actual_duration), 0, None
            )
            # Moderate response_time increase
            df.loc[start_idx:end_idx - 1, "response_time_ms"] = np.clip(
                rt_baseline * rng.uniform(1.20, 1.50, actual_duration), 0, 10000
            )

            df.loc[start_idx:end_idx - 1, "is_anomaly"]   = 1
            df.loc[start_idx:end_idx - 1, "anomaly_type"] = "error_cascade"
            injected_mask[start_idx:end_idx] = True
            events_injected += 1

        # ── Disk Bottleneck: 4–12 readings (1–3 hours) ───────────────────────
        elif atype == "disk_bottleneck":
            duration = int(rng.integers(4, 13))
            end_idx  = min(start_idx + duration, n)
            actual_duration = end_idx - start_idx

            disk_baseline = df.loc[start_idx, "disk_io_mbps"]
            rt_baseline   = df.loc[start_idx, "response_time_ms"]

            # Sustained plateau (constant elevated level), not a spike
            plateau_mult = rng.uniform(2.5, 4.0)
            df.loc[start_idx:end_idx - 1, "disk_io_mbps"] = np.clip(
                disk_baseline * plateau_mult + rng.normal(0, disk_baseline * 0.05, actual_duration),
                0, 1000
            )
            # Moderate response_time increase (second metric to make it detectable via combination)
            df.loc[start_idx:end_idx - 1, "response_time_ms"] = np.clip(
                rt_baseline * rng.uniform(1.15, 1.40, actual_duration), 0, 10000
            )

            df.loc[start_idx:end_idx - 1, "is_anomaly"]   = 1
            df.loc[start_idx:end_idx - 1, "anomaly_type"] = "disk_bottleneck"
            injected_mask[start_idx:end_idx] = True
            events_injected += 1

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR: Generate the full 15-server dataset
# ─────────────────────────────────────────────────────────────────────────────
def generate_full_dataset(
    master_seed: int = 42,
    days: int = 30,
    interval_minutes: int = 15,
    output_path: str = "../data/raw/infra_logs.csv",
    target_anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Generate the complete synthetic infrastructure log dataset.

    Parameters
    ----------
    master_seed : int
        Master random seed. Each server gets its own seeded RNG derived
        from master_seed + server_index for reproducibility.
    days : int
        Number of days of history to simulate (default 30).
    interval_minutes : int
        Granularity in minutes (default 15 → 96 readings/day).
    output_path : str
        Where to save the CSV (relative to the calling notebook's location).
    target_anomaly_rate : float
        Approximate fraction of rows to mark as anomalous (default 5%).

    Returns
    -------
    pd.DataFrame
        Full dataset sorted by (timestamp, server_id).
    """
    # Generate timestamps: 30 days × 96 readings/day = 2,880 per server
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=days * (1440 // interval_minutes),
                               freq=f"{interval_minutes}min")

    all_dfs = []

    for i, (server_id, service_type) in enumerate(FLEET):
        # Each server has its own seeded RNG → same master_seed always produces same dataset
        server_rng = np.random.default_rng(master_seed + i)

        # Generate normal baseline
        server_df = generate_baseline_metrics(server_id, service_type, timestamps, server_rng)

        # Inject anomalies (labels stored but not used for modeling)
        server_df = inject_anomalies(server_df, server_rng, target_anomaly_rate)

        all_dfs.append(server_df)
        print(f"  [OK] {server_id:<20} ({service_type:<18}) | "
              f"anomaly rate: {server_df['is_anomaly'].mean()*100:.1f}%")

    full_df = pd.concat(all_dfs, ignore_index=True).sort_values(
        ["timestamp", "server_id"]
    ).reset_index(drop=True)

    # Save to CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output, index=False)
    print(f"\n[SAVED] Dataset saved to: {output.resolve()}")

    return full_df


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION HELPER (called from Notebook 01)
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a verification summary of the generated dataset.
    Confirms the anomaly injection hit the 4–6% target and
    is distributed across all types and server types.
    """
    print("-" * 60)
    print("DATASET SUMMARY")
    print("-" * 60)
    print(f"Total rows          : {len(df):,}")
    print(f"Total servers       : {df['server_id'].nunique()}")
    print(f"Date range          : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Overall anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")
    print()

    print("-- Anomaly rate by type --------------------------------")
    type_summary = df.groupby("anomaly_type").agg(
        count=("is_anomaly", "sum"),
        pct=("is_anomaly", lambda x: x.mean() * 100)
    ).round(2)
    print(type_summary.to_string())
    print()

    print("-- Anomaly rate by service type ------------------------")
    svc_summary = df.groupby("service_type").agg(
        total_rows=("is_anomaly", "count"),
        anomaly_rows=("is_anomaly", "sum"),
        anomaly_pct=("is_anomaly", lambda x: x.mean() * 100)
    ).round(2)
    print(svc_summary.to_string())
    print()

    print("-- Anomaly rate by server ------------------------------")
    srv_summary = df.groupby(["server_id", "service_type"]).agg(
        anomaly_pct=("is_anomaly", lambda x: x.mean() * 100)
    ).round(2)
    print(srv_summary.to_string())
    print("-" * 60)
