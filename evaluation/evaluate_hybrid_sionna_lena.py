from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.home() / "ndt_modeling_benchmark"

TRAJ1 = ROOT / "models/hybrid_sionna_lena/powder_run1_trajectory.csv"
TRAJ2 = ROOT / "models/hybrid_sionna_lena/powder_run2_trajectory.csv"

TRACE1 = ROOT / "results/hybrid_sionna_lena_run1_500ms_raw.csv"
TRACE2 = ROOT / "results/hybrid_sionna_lena_run2_500ms_raw.csv"

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def load_trace(path):
    df = pd.read_csv(path)

    # Handle trace with or without CSV header
    if "time_s" not in df.columns:
        df = pd.read_csv(
            path,
            names=[
                "time_s", "frame", "subframe", "slot",
                "bwp_id", "cell_id", "duration_s", "rx_power_dbm"
            ],
            header=None,
        )

    for c in ["time_s", "rx_power_dbm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["time_s", "rx_power_dbm"]).sort_values("time_s")


def load_trajectory(path):
    df = pd.read_csv(path)

    # Find simulation-time column
    time_candidates = ["sim_time_s", "time_s"]
    tcol = next((c for c in time_candidates if c in df.columns), None)

    if tcol is None:
        raise RuntimeError(f"No simulation-time column found in {path}")

    # Find physical RSRP column
    rsrp_candidates = [
        "rsrp_real_dbm",
        "RSRP_real_dbm",
        "rsrp_dbm"
    ]
    rcol = next((c for c in rsrp_candidates if c in df.columns), None)

    if rcol is None:
        raise RuntimeError(
            f"No physical RSRP column found in {path}. "
            f"Columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    if "sample_id" in df.columns:
        out["sample_id"] = df["sample_id"]
    else:
        out["sample_id"] = np.arange(len(df))

    out["time_s"] = pd.to_numeric(df[tcol], errors="coerce")
    out["rsrp_real_dbm"] = pd.to_numeric(df[rcol], errors="coerce")

    return out.dropna().sort_values("time_s")


def align_nearest(meas, trace):
    """
    Match every POWDER measurement to the temporally nearest ns-3
    RxDataTrace event.

    This avoids interpolation of the physical-model output.
    """
    left = meas.sort_values("time_s").copy()

    right = trace[["time_s", "rx_power_dbm"]].copy()
    right = right.rename(columns={
        "time_s": "trace_time_s",
        "rx_power_dbm": "rx_power_dbm_raw"
    }).sort_values("trace_time_s")

    aligned = pd.merge_asof(
        left,
        right,
        left_on="time_s",
        right_on="trace_time_s",
        direction="nearest",
    )

    aligned["time_error_ms"] = (
        aligned["trace_time_s"] - aligned["time_s"]
    ).abs() * 1000.0

    return aligned


def calibration_delta(train):
    """
    One scalar additive calibration only.
    Median minimizes absolute error on training data.
    """
    return float(
        np.median(
            train["rsrp_real_dbm"].to_numpy()
            - train["rx_power_dbm_raw"].to_numpy()
        )
    )


def evaluate(train, test):
    delta = calibration_delta(train)

    out = test.copy()
    out["calibration_delta_db"] = delta
    out["rx_power_dbm_calibrated"] = (
        out["rx_power_dbm_raw"] + delta
    )

    error = (
        out["rsrp_real_dbm"]
        - out["rx_power_dbm_calibrated"]
    )

    raw_error = (
        out["rsrp_real_dbm"]
        - out["rx_power_dbm_raw"]
    )

    metrics = {
        "delta_db": delta,
        "n_test": int(len(out)),
        "mae_db": float(np.mean(np.abs(error))),
        "rmse_db": float(np.sqrt(np.mean(error ** 2))),
        "bias_db": float(np.mean(
            out["rx_power_dbm_calibrated"]
            - out["rsrp_real_dbm"]
        )),
        "raw_mae_db": float(np.mean(np.abs(raw_error))),
        "median_time_error_ms": float(
            out["time_error_ms"].median()
        ),
        "p95_time_error_ms": float(
            out["time_error_ms"].quantile(0.95)
        ),
        "max_time_error_ms": float(
            out["time_error_ms"].max()
        ),
    }

    return out, metrics


print("Loading trajectories...")
m1 = load_trajectory(TRAJ1)
m2 = load_trajectory(TRAJ2)

print("Run1 measurements:", len(m1))
print("Run2 measurements:", len(m2))

print("\nLoading ns-3/Sionna traces...")
t1 = load_trace(TRACE1)
t2 = load_trace(TRACE2)

print("Run1 trace events:", len(t1))
print("Run2 trace events:", len(t2))

print("\nTemporal alignment...")
a1 = align_nearest(m1, t1)
a2 = align_nearest(m2, t2)

a1.to_csv(OUT / "hybrid_run1_aligned_raw.csv", index=False)
a2.to_csv(OUT / "hybrid_run2_aligned_raw.csv", index=False)

# ---------------------------------------------------------
# Cross-run evaluation
# ---------------------------------------------------------

# Train/calibrate on Run1 -> test Run2
r12, met12 = evaluate(a1, a2)
r12["direction"] = "R1->R2"

# Train/calibrate on Run2 -> test Run1
r21, met21 = evaluate(a2, a1)
r21["direction"] = "R2->R1"

r12.to_csv(
    OUT / "hybrid_run1_to_run2_predictions.csv",
    index=False
)

r21.to_csv(
    OUT / "hybrid_run2_to_run1_predictions.csv",
    index=False
)

results = pd.DataFrame([
    {
        "direction": "R1->R2",
        **met12
    },
    {
        "direction": "R2->R1",
        **met21
    },
])

results.to_csv(
    OUT / "hybrid_crossrun_results.csv",
    index=False
)

# ---------------------------------------------------------
# Resource measurements already obtained experimentally
# ---------------------------------------------------------

resources = pd.DataFrame([
    {
        "run": "Run1",
        "sim_duration_s": 703.0,
        "wall_time_s": 46*60 + 55.88,
        "cpu_percent": 1047.0,
        "peak_rss_kb": 8715312,
        "peak_rss_gib": 8715312 / 1024 / 1024,
        "update_period_ms": 500,
    },
    {
        "run": "Run2",
        "sim_duration_s": 481.0,
        "wall_time_s": 25*60 + 39.71,
        "cpu_percent": 1258.0,
        "peak_rss_kb": 8607740,
        "peak_rss_gib": 8607740 / 1024 / 1024,
        "update_period_ms": 500,
    },
])

resources.to_csv(
    OUT / "hybrid_resources.csv",
    index=False
)

summary = {
    "model": "Sionna RT + ns-3/5G-LENA",
    "update_period_ms": 500,
    "R1_to_R2": met12,
    "R2_to_R1": met21,
    "resources": {
        "run1_wall_s": 46*60 + 55.88,
        "run2_wall_s": 25*60 + 39.71,
        "total_wall_s": (46*60 + 55.88) + (25*60 + 39.71),
        "mean_cpu_percent": (1047 + 1258) / 2,
        "mean_peak_rss_gib": float(
            resources["peak_rss_gib"].mean()
        ),
    }
}

with open(OUT / "hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n==============================")
print(" SIONNA RT + 5G-LENA RESULTS")
print("==============================")

print("\nR1 -> R2")
for k, v in met12.items():
    print(f"{k:25s}: {v}")

print("\nR2 -> R1")
for k, v in met21.items():
    print(f"{k:25s}: {v}")

print("\nResources")
print(resources.to_string(index=False))

print("\nSaved:")
print(" - hybrid_run1_aligned_raw.csv")
print(" - hybrid_run2_aligned_raw.csv")
print(" - hybrid_run1_to_run2_predictions.csv")
print(" - hybrid_run2_to_run1_predictions.csv")
print(" - hybrid_crossrun_results.csv")
print(" - hybrid_resources.csv")
print(" - hybrid_summary.json")
