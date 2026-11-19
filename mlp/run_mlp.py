import json
import os
import time
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from pyproj import Transformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


DATA = Path("data/processed/powder_benchmark_with_los.csv")
CONFIG = Path("configs/analytical_3gpp.json")
OUT = Path("results/ai/mlp_seed42")

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# ------------------------------------------------------------
# RESOURCE MONITOR
# ------------------------------------------------------------

def monitor_process(stop_event, rows, interval=0.02):
    p = psutil.Process(os.getpid())
    p.cpu_percent(None)
    t0 = time.perf_counter()

    while not stop_event.is_set():
        rows.append({
            "elapsed_s": time.perf_counter() - t0,
            "cpu_percent": p.cpu_percent(None),
            "ram_mb": p.memory_info().rss / (1024**2),
            "system_ram_percent": psutil.virtual_memory().percent
        })
        time.sleep(interval)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA)

with open(CONFIG) as f:
    cfg = json.load(f)

required = [
    "latitude",
    "longitude",
    "height_m",
    "ray_distance_2d_m",
    "rsrp_real_dbm",
    "run"
]

df = df.dropna(subset=required).copy()

# ------------------------------------------------------------
# WGS84 -> LOCAL METRIC COORDINATES
# UTM Zone 12N for Salt Lake City region
# ------------------------------------------------------------

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32612",
    always_xy=True
)

x, y = transformer.transform(
    df["longitude"].to_numpy(),
    df["latitude"].to_numpy()
)

gnb_x, gnb_y = transformer.transform(
    cfg["gnb_longitude"],
    cfg["gnb_latitude"]
)

df["x_m"] = x - gnb_x
df["y_m"] = y - gnb_y

# Height relative to training-run median.
# This avoids feeding ~1450 m absolute altitude unnecessarily.
run1_height_ref = float(
    df.loc[df["run"] == "run1", "height_m"].median()
)

df["height_rel_m"] = (
    df["height_m"] - run1_height_ref
)

FEATURES = [
    "x_m",
    "y_m",
    "height_rel_m",
    "ray_distance_2d_m"
]

TARGET = "rsrp_real_dbm"

train = df[df["run"] == "run1"].copy()
test = df[df["run"] == "run2"].copy()

X_train = train[FEATURES].to_numpy()
y_train = train[TARGET].to_numpy()

X_test = test[FEATURES].to_numpy()
y_test = test[TARGET].to_numpy()

# ------------------------------------------------------------
# NORMALIZATION
# Fit ONLY on Run1
# ------------------------------------------------------------

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_s = x_scaler.fit_transform(X_train)
X_test_s = x_scaler.transform(X_test)

y_train_s = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation="relu",
    solver="adam",

    learning_rate_init=1e-3,

    max_iter=1000,

    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=40,

    random_state=SEED,

    batch_size=64,

    verbose=False
)

# ============================================================
# TRAINING RESOURCE PROFILE
# ============================================================

train_resources = []
stop_train = threading.Event()

thread = threading.Thread(
    target=monitor_process,
    args=(stop_train, train_resources),
    daemon=True
)

thread.start()

train_wall_start = time.perf_counter()
train_cpu_start = time.process_time()

model.fit(
    X_train_s,
    y_train_s
)

train_runtime_s = (
    time.perf_counter() - train_wall_start
)

train_cpu_s = (
    time.process_time() - train_cpu_start
)

stop_train.set()
thread.join()

train_resources = pd.DataFrame(
    train_resources
)

train_resources.to_csv(
    OUT / "training_resources.csv",
    index=False
)

# ------------------------------------------------------------
# TRAIN PREDICTION
# ------------------------------------------------------------

train_pred_s = model.predict(
    X_train_s
)

train_pred = y_scaler.inverse_transform(
    train_pred_s.reshape(-1, 1)
).ravel()

train_mae = mean_absolute_error(
    y_train,
    train_pred
)

# ============================================================
# TRUE TEST INFERENCE RUNTIME
# ============================================================

# Warm-up
_ = model.predict(
    X_test_s[:10]
)

infer_cpu_start = time.process_time()
infer_wall_start = time.perf_counter()

test_pred_s = model.predict(
    X_test_s
)

infer_runtime_s = (
    time.perf_counter() - infer_wall_start
)

infer_cpu_s = (
    time.process_time() - infer_cpu_start
)

test_pred = y_scaler.inverse_transform(
    test_pred_s.reshape(-1, 1)
).ravel()

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

# ============================================================
# INFERENCE RESOURCE PROFILE
# Repeated only to obtain a useful time series.
# ============================================================

infer_resources = []
stop_infer = threading.Event()

thread = threading.Thread(
    target=monitor_process,
    args=(stop_infer, infer_resources),
    daemon=True
)

thread.start()

profile_start = time.perf_counter()
profile_iterations = 0

while time.perf_counter() - profile_start < 2.0:
    _ = model.predict(X_test_s)
    profile_iterations += 1

stop_infer.set()
thread.join()

infer_resources = pd.DataFrame(
    infer_resources
)

infer_resources.to_csv(
    OUT / "inference_resources.csv",
    index=False
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

train["split"] = "train"
train["rsrp_mlp_dbm"] = train_pred
train["abs_error_db"] = np.abs(
    train[TARGET] - train_pred
)

test["split"] = "test"
test["rsrp_mlp_dbm"] = test_pred
test["abs_error_db"] = np.abs(
    test[TARGET] - test_pred
)

pred = pd.concat(
    [train, test],
    ignore_index=True
)

prediction_cols = [
    "sample_id",
    "run",
    "split",
    "timestamp",
    "latitude",
    "longitude",
    "x_m",
    "y_m",
    "height_m",
    "height_rel_m",
    "ray_distance_2d_m",
    "rsrp_real_dbm",
    "rsrp_mlp_dbm",
    "abs_error_db"
]

pred[prediction_cols].to_csv(
    OUT / "predictions.csv",
    index=False
)

# ------------------------------------------------------------
# TRAINING HISTORY
# ------------------------------------------------------------

history = pd.DataFrame({
    "epoch": np.arange(
        1,
        len(model.loss_curve_) + 1
    ),
    "training_loss": model.loss_curve_
})

history.to_csv(
    OUT / "training_history.csv",
    index=False
)

# ------------------------------------------------------------
# SAVE MODEL + SCALERS
# ------------------------------------------------------------

joblib.dump(
    model,
    OUT / "mlp_model.joblib"
)

joblib.dump(
    x_scaler,
    OUT / "x_scaler.joblib"
)

joblib.dump(
    y_scaler,
    OUT / "y_scaler.joblib"
)

# ============================================================
# METRICS
# ============================================================

metrics = {
    "model": "MLP_64_64_32",
    "seed": SEED,

    "features": FEATURES,

    "train_run": "run1",
    "test_run": "run2",

    "n_train": int(len(train)),
    "n_test": int(len(test)),

    "epochs": int(model.n_iter_),

    "train_mae_db": float(train_mae),
    "test_mae_db": float(test_mae),

    "training_runtime_s": float(
        train_runtime_s
    ),

    "training_cpu_time_s": float(
        train_cpu_s
    ),

    "inference_runtime_s": float(
        infer_runtime_s
    ),

    "inference_cpu_time_s": float(
        infer_cpu_s
    ),

    "inference_time_per_sample_us": float(
        infer_runtime_s / len(test) * 1e6
    ),

    "profile_iterations": int(
        profile_iterations
    ),

    "training_cpu_mean_percent": (
        float(
            train_resources[
                "cpu_percent"
            ].mean()
        )
        if len(train_resources)
        else None
    ),

    "training_cpu_peak_percent": (
        float(
            train_resources[
                "cpu_percent"
            ].max()
        )
        if len(train_resources)
        else None
    ),

    "training_ram_peak_mb": (
        float(
            train_resources[
                "ram_mb"
            ].max()
        )
        if len(train_resources)
        else None
    ),

    "inference_cpu_mean_percent": (
        float(
            infer_resources[
                "cpu_percent"
            ].mean()
        )
        if len(infer_resources)
        else None
    ),

    "inference_cpu_peak_percent": (
        float(
            infer_resources[
                "cpu_percent"
            ].max()
        )
        if len(infer_resources)
        else None
    ),

    "inference_ram_mean_mb": (
        float(
            infer_resources[
                "ram_mb"
            ].mean()
        )
        if len(infer_resources)
        else None
    ),

    "inference_ram_peak_mb": (
        float(
            infer_resources[
                "ram_mb"
            ].max()
        )
        if len(infer_resources)
        else None
    )
}

with open(
    OUT / "metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )

print("=" * 72)
print("MLP RSRP BASELINE")
print("=" * 72)

print("Features:", FEATURES)
print()

print(
    f"Train samples: {len(train)}"
)

print(
    f"Test samples : {len(test)}"
)

print(
    f"Epochs       : {model.n_iter_}"
)

print()

print(
    f"Train MAE: {train_mae:.3f} dB"
)

print(
    f"TEST MAE : {test_mae:.3f} dB"
)

print()

print(
    f"Training runtime: "
    f"{train_runtime_s:.3f} s"
)

print(
    f"Inference runtime: "
    f"{infer_runtime_s*1000:.6f} ms"
)

print(
    f"Inference/sample: "
    f"{infer_runtime_s/len(test)*1e6:.6f} us"
)

print()

print(
    "Training RAM peak:",
    metrics["training_ram_peak_mb"],
    "MB"
)

print(
    "Inference RAM mean/peak:",
    metrics["inference_ram_mean_mb"],
    "/",
    metrics["inference_ram_peak_mb"],
    "MB"
)

print()
print("Saved:")
print(OUT / "predictions.csv")
print(OUT / "training_history.csv")
print(OUT / "training_resources.csv")
print(OUT / "inference_resources.csv")
print(OUT / "metrics.json")
print(OUT / "mlp_model.joblib")
