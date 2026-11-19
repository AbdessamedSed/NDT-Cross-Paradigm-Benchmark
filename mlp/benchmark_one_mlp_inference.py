import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from pyproj import Transformer


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python benchmark_one_mlp_inference.py MODEL_DIR"
    )

MODEL_DIR = Path(sys.argv[1])

DATA = Path(
    "data/processed/powder_benchmark_with_los.csv"
)

CONFIG = Path(
    "configs/analytical_3gpp.json"
)

WARMUPS = 100
REPEATS = 1000


with open(
    MODEL_DIR / "metrics.json"
) as f:
    meta = json.load(f)

with open(CONFIG) as f:
    cfg = json.load(f)


model = joblib.load(
    MODEL_DIR / "model.joblib"
)

x_scaler = joblib.load(
    MODEL_DIR / "x_scaler.joblib"
)

y_scaler = joblib.load(
    MODEL_DIR / "y_scaler.joblib"
)


df = pd.read_csv(DATA)

test_run = meta["test_run"]

test = df[
    df["run"] == test_run
].copy()


transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32612",
    always_xy=True
)

x, y = transformer.transform(
    test["longitude"].to_numpy(),
    test["latitude"].to_numpy()
)

gnb_x, gnb_y = transformer.transform(
    cfg["gnb_longitude"],
    cfg["gnb_latitude"]
)

test["x_m"] = x - gnb_x
test["y_m"] = y - gnb_y

test["height_rel_m"] = (
    test["height_m"]
    -
    meta["height_reference_m"]
)

features = meta["features"]

X = test[features].to_numpy()

X_s = x_scaler.transform(X)


# --------------------------------------------
# PROCESS RAM BASELINE
# --------------------------------------------

proc = psutil.Process()

ram_baseline_mb = (
    proc.memory_info().rss
    /
    (1024**2)
)


# --------------------------------------------
# WARM-UP
# --------------------------------------------

for _ in range(WARMUPS):

    pred_s = model.predict(X_s)

    _ = y_scaler.inverse_transform(
        pred_s.reshape(-1, 1)
    )


# --------------------------------------------
# CLEAN TIMING
# --------------------------------------------

timings = []

peak_ram_mb = ram_baseline_mb


for i in range(REPEATS):

    t0 = time.perf_counter_ns()

    pred_s = model.predict(X_s)

    pred = y_scaler.inverse_transform(
        pred_s.reshape(-1, 1)
    ).ravel()

    t1 = time.perf_counter_ns()

    timings.append(
        t1 - t0
    )

    rss = (
        proc.memory_info().rss
        /
        (1024**2)
    )

    peak_ram_mb = max(
        peak_ram_mb,
        rss
    )


timings = np.asarray(
    timings,
    dtype=float
)

median_ms = float(
    np.median(timings) / 1e6
)

mean_ms = float(
    np.mean(timings) / 1e6
)

std_ms = float(
    np.std(timings, ddof=1) / 1e6
)

per_sample_us = (
    median_ms * 1000
    /
    len(test)
)


# --------------------------------------------
# SAVE PER MODEL
# --------------------------------------------

pd.DataFrame({
    "repeat":
        np.arange(REPEATS),

    "runtime_ns":
        timings,

    "runtime_ms":
        timings / 1e6
}).to_csv(
    MODEL_DIR /
    "clean_inference_timings.csv",
    index=False
)


summary = {
    "experiment_id":
        meta["experiment_id"],

    "architecture":
        meta["architecture"],

    "seed":
        meta["seed"],

    "train_run":
        meta["train_run"],

    "test_run":
        meta["test_run"],

    "n_test":
        len(test),

    "test_mae_db":
        meta["test_mae_db"],

    "inference_batch_median_ms":
        median_ms,

    "inference_batch_mean_ms":
        mean_ms,

    "inference_batch_std_ms":
        std_ms,

    "inference_per_sample_median_us":
        per_sample_us,

    "ram_baseline_mb":
        ram_baseline_mb,

    "ram_peak_mb":
        peak_ram_mb,

    "delta_ram_mb":
        peak_ram_mb -
        ram_baseline_mb
}


with open(
    MODEL_DIR /
    "clean_inference_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print(
    json.dumps(
        summary
    )
)
