import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed/powder_benchmark_with_los.csv")
CONFIG = Path("configs/analytical_3gpp.json")
OUT = Path("results/analytical/final_stats")
OUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAPS = 2000
BLOCK_LEN = 20
TIMING_REPEATS = 5000
WARMUPS = 200
SEED = 42

rng = np.random.default_rng(SEED)


def umi_los(d2d, fc_ghz, h_bs, h_ut):
    d2d = np.maximum(np.asarray(d2d, dtype=float), 10.0)

    d3d = np.sqrt(
        d2d**2 +
        (h_bs - h_ut)**2
    )

    h_e = 1.0
    h_bs_eff = h_bs - h_e
    h_ut_eff = h_ut - h_e

    fc_hz = fc_ghz * 1e9
    c = 3e8

    d_bp = (
        4.0 *
        h_bs_eff *
        h_ut_eff *
        fc_hz / c
    )

    pl1 = (
        32.4
        + 21*np.log10(d3d)
        + 20*np.log10(fc_ghz)
    )

    pl2 = (
        32.4
        + 40*np.log10(d3d)
        + 20*np.log10(fc_ghz)
        - 9.5*np.log10(
            d_bp**2 +
            (h_bs-h_ut)**2
        )
    )

    return np.where(
        d2d <= d_bp,
        pl1,
        pl2
    )


def block_bootstrap_mae(errors, n_boot=2000, block_len=20):
    errors = np.asarray(errors)
    n = len(errors)

    maes = []

    for _ in range(n_boot):
        sample = []

        while len(sample) < n:
            start = rng.integers(
                0,
                max(1, n-block_len+1)
            )

            sample.extend(
                errors[
                    start:start+block_len
                ]
            )

        sample = np.asarray(sample[:n])

        maes.append(
            np.mean(np.abs(sample))
        )

    maes = np.asarray(maes)

    return (
        float(np.mean(maes)),
        float(np.percentile(maes, 2.5)),
        float(np.percentile(maes, 97.5))
    )


with open(CONFIG) as f:
    cfg = json.load(f)

df = pd.read_csv(DATA)

fc = cfg["frequency_ghz"]
h_bs = cfg["h_bs_m"]
h_ut = cfg["h_ut_m"]

all_results = []
all_predictions = []
all_timings = []

splits = [
    ("run1", "run2"),
    ("run2", "run1"),
]

for train_run, test_run in splits:

    train = df[df["run"] == train_run].copy()
    test = df[df["run"] == test_run].copy()

    train_pl = umi_los(
        train["ray_distance_2d_m"].to_numpy(),
        fc,
        h_bs,
        h_ut
    )

    test_d = test[
        "ray_distance_2d_m"
    ].to_numpy()

    test_pl = umi_los(
        test_d,
        fc,
        h_bs,
        h_ut
    )

    # Calibration using train only
    p_ref = float(
        np.median(
            train["rsrp_real_dbm"].to_numpy()
            +
            train_pl
        )
    )

    pred = p_ref - test_pl

    real = test[
        "rsrp_real_dbm"
    ].to_numpy()

    errors = pred - real

    mae = float(
        np.mean(np.abs(errors))
    )

    boot_mean, ci_low, ci_high = (
        block_bootstrap_mae(
            errors,
            BOOTSTRAPS,
            BLOCK_LEN
        )
    )

    # --------------------------------------------
    # CLEAN INFERENCE TIMING
    # --------------------------------------------

    for _ in range(WARMUPS):
        pl = umi_los(
            test_d,
            fc,
            h_bs,
            h_ut
        )
        _ = p_ref - pl

    times_ns = []

    for _ in range(TIMING_REPEATS):

        t0 = time.perf_counter_ns()

        pl = umi_los(
            test_d,
            fc,
            h_bs,
            h_ut
        )

        _ = p_ref - pl

        t1 = time.perf_counter_ns()

        times_ns.append(
            t1 - t0
        )

    times_ns = np.asarray(
        times_ns,
        dtype=float
    )

    median_ms = float(
        np.median(times_ns) / 1e6
    )

    mean_ms = float(
        np.mean(times_ns) / 1e6
    )

    std_ms = float(
        np.std(times_ns, ddof=1) / 1e6
    )

    per_sample_us = (
        median_ms * 1000.0
        /
        len(test)
    )

    result = {
        "train_run": train_run,
        "test_run": test_run,

        "n_train": len(train),
        "n_test": len(test),

        "p_ref_dbm": p_ref,

        "test_mae_db": mae,

        "bootstrap_mae_mean_db":
            boot_mean,

        "mae_ci95_low_db":
            ci_low,

        "mae_ci95_high_db":
            ci_high,

        "inference_batch_median_ms":
            median_ms,

        "inference_batch_mean_ms":
            mean_ms,

        "inference_batch_std_ms":
            std_ms,

        "inference_per_sample_median_us":
            per_sample_us
    }

    all_results.append(result)

    out = test.copy()

    out["split"] = (
        f"{train_run}_to_{test_run}"
    )

    out["rsrp_3gpp_dbm"] = pred
    out["signed_error_db"] = errors
    out["abs_error_db"] = np.abs(errors)

    all_predictions.append(out)

    for i, t in enumerate(times_ns):

        all_timings.append({
            "train_run": train_run,
            "test_run": test_run,
            "repeat": i,
            "runtime_ns": t,
            "runtime_ms": t / 1e6
        })


results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUT / "summary.csv",
    index=False
)

pd.concat(
    all_predictions,
    ignore_index=True
).to_csv(
    OUT / "predictions.csv",
    index=False
)

pd.DataFrame(
    all_timings
).to_csv(
    OUT / "inference_timings.csv",
    index=False
)

print("="*88)
print("FINAL 3GPP STATISTICAL BENCHMARK")
print("="*88)

print(
    results[
        [
            "train_run",
            "test_run",
            "test_mae_db",
            "mae_ci95_low_db",
            "mae_ci95_high_db",
            "inference_batch_median_ms",
            "inference_per_sample_median_us"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUT / "summary.csv")
print(OUT / "predictions.csv")
print(OUT / "inference_timings.csv")
