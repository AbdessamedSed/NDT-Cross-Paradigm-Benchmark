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


# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path("data/processed/powder_benchmark_with_los.csv")
CONFIG = Path("configs/analytical_3gpp.json")

ROOT_OUT = Path("results/ai/mlp_benchmark")
ROOT_OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [
    7,
    21,
    42,
    77,
    123,
    202,
    314,
    512,
    777,
    1024
]

ARCHITECTURES = {
    "mlp_64_64_32": (64, 64, 32),
    "mlp_128_128_64": (128, 128, 64),
    "mlp_256_128_64": (256, 128, 64),
}

SPLITS = [
    ("run1", "run2"),
    ("run2", "run1"),
]

MAX_ITER = 3000
EARLY_STOPPING_PATIENCE = 60
BATCH_SIZE = 64

# Resource sampling
RESOURCE_INTERVAL_S = 0.02

# Repeated inference is ONLY for resource-profile time series.
# True inference runtime is measured separately.
INFERENCE_PROFILE_DURATION_S = 1.0


# ============================================================
# RESOURCE MONITOR
# ============================================================

def monitor_resources(stop_event, rows, interval=RESOURCE_INTERVAL_S):
    proc = psutil.Process(os.getpid())

    proc.cpu_percent(None)

    t0 = time.perf_counter()

    while not stop_event.is_set():

        rows.append({
            "elapsed_s": time.perf_counter() - t0,

            "process_cpu_percent":
                proc.cpu_percent(None),

            "process_ram_mb":
                proc.memory_info().rss / (1024 ** 2),

            "system_ram_percent":
                psutil.virtual_memory().percent
        })

        time.sleep(interval)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)

with open(CONFIG) as f:
    cfg = json.load(f)


required = [
    "sample_id",
    "run",
    "timestamp",
    "latitude",
    "longitude",
    "height_m",
    "ray_distance_2d_m",
    "rsrp_real_dbm"
]

df = df.dropna(
    subset=required
).copy()


# ============================================================
# GPS -> LOCAL METRIC COORDINATES
# ============================================================

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


FEATURES_BASE = [
    "x_m",
    "y_m",
    "height_rel_m",
    "ray_distance_2d_m"
]

TARGET = "rsrp_real_dbm"


# ============================================================
# GLOBAL STORAGE
# ============================================================

summary_rows = []


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

total_runs = (
    len(SEEDS)
    * len(ARCHITECTURES)
    * len(SPLITS)
)

run_counter = 0


for train_run, test_run in SPLITS:

    # --------------------------------------------------------
    # Split-specific height reference
    # Fit using TRAIN only.
    # --------------------------------------------------------

    height_reference = float(
        df.loc[
            df["run"] == train_run,
            "height_m"
        ].median()
    )

    df_split = df.copy()

    df_split["height_rel_m"] = (
        df_split["height_m"]
        -
        height_reference
    )


    train_df = df_split[
        df_split["run"] == train_run
    ].copy()

    test_df = df_split[
        df_split["run"] == test_run
    ].copy()


    for arch_name, hidden_layers in ARCHITECTURES.items():

        for seed in SEEDS:

            run_counter += 1

            experiment_id = (
                f"{train_run}_to_{test_run}"
                f"__{arch_name}"
                f"__seed_{seed}"
            )

            print()
            print("=" * 80)
            print(
                f"[{run_counter}/{total_runs}] "
                f"{experiment_id}"
            )
            print("=" * 80)

            OUT = ROOT_OUT / experiment_id
            OUT.mkdir(
                parents=True,
                exist_ok=True
            )

            # ====================================================
            # DATA ARRAYS
            # ====================================================

            X_train = train_df[
                FEATURES_BASE
            ].to_numpy()

            y_train = train_df[
                TARGET
            ].to_numpy()

            X_test = test_df[
                FEATURES_BASE
            ].to_numpy()

            y_test = test_df[
                TARGET
            ].to_numpy()


            # ====================================================
            # NORMALIZATION
            # TRAIN ONLY
            # ====================================================

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X_train_s = x_scaler.fit_transform(
                X_train
            )

            X_test_s = x_scaler.transform(
                X_test
            )

            y_train_s = y_scaler.fit_transform(
                y_train.reshape(-1, 1)
            ).ravel()


            # ====================================================
            # MODEL
            # ====================================================

            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,

                activation="relu",
                solver="adam",

                learning_rate_init=1e-3,

                max_iter=MAX_ITER,

                early_stopping=True,
                validation_fraction=0.15,

                n_iter_no_change=
                    EARLY_STOPPING_PATIENCE,

                batch_size=BATCH_SIZE,

                random_state=seed,

                verbose=False
            )


            # ====================================================
            # TRAINING RESOURCE PROFILE
            # ====================================================

            training_resources = []

            stop_train_monitor = threading.Event()

            monitor_thread = threading.Thread(
                target=monitor_resources,
                args=(
                    stop_train_monitor,
                    training_resources
                ),
                daemon=True
            )

            monitor_thread.start()

            train_wall_start = time.perf_counter()
            train_cpu_start = time.process_time()

            model.fit(
                X_train_s,
                y_train_s
            )

            training_runtime_s = (
                time.perf_counter()
                -
                train_wall_start
            )

            training_cpu_time_s = (
                time.process_time()
                -
                train_cpu_start
            )

            stop_train_monitor.set()
            monitor_thread.join()

            training_resources = pd.DataFrame(
                training_resources
            )

            training_resources.to_csv(
                OUT / "training_resources.csv",
                index=False
            )


            # ====================================================
            # TRAIN METRICS
            # ====================================================

            train_pred_scaled = model.predict(
                X_train_s
            )

            train_pred = y_scaler.inverse_transform(
                train_pred_scaled.reshape(-1, 1)
            ).ravel()

            train_mae = mean_absolute_error(
                y_train,
                train_pred
            )


            # ====================================================
            # TEST INFERENCE - TRUE RUNTIME
            # ====================================================

            # Warm-up outside timing
            _ = model.predict(
                X_test_s[:10]
            )

            infer_cpu_start = time.process_time()
            infer_wall_start = time.perf_counter()

            test_pred_scaled = model.predict(
                X_test_s
            )

            inference_runtime_s = (
                time.perf_counter()
                -
                infer_wall_start
            )

            inference_cpu_time_s = (
                time.process_time()
                -
                infer_cpu_start
            )

            test_pred = y_scaler.inverse_transform(
                test_pred_scaled.reshape(-1, 1)
            ).ravel()


            # ====================================================
            # TEST METRICS
            # ====================================================

            test_mae = mean_absolute_error(
                y_test,
                test_pred
            )

            signed_error = (
                test_pred
                -
                y_test
            )

            abs_error = np.abs(
                signed_error
            )


            # ====================================================
            # INFERENCE RESOURCE PROFILE
            # repeated workload only for resource time-series
            # ====================================================

            inference_resources = []

            stop_infer_monitor = threading.Event()

            monitor_thread = threading.Thread(
                target=monitor_resources,
                args=(
                    stop_infer_monitor,
                    inference_resources
                ),
                daemon=True
            )

            monitor_thread.start()

            profile_start = time.perf_counter()
            profile_iterations = 0

            while (
                time.perf_counter()
                -
                profile_start
                <
                INFERENCE_PROFILE_DURATION_S
            ):
                _ = model.predict(
                    X_test_s
                )

                profile_iterations += 1

            stop_infer_monitor.set()
            monitor_thread.join()

            inference_resources = pd.DataFrame(
                inference_resources
            )

            inference_resources.to_csv(
                OUT / "inference_resources.csv",
                index=False
            )


            # ====================================================
            # PREDICTIONS
            # ====================================================

            train_output = train_df.copy()

            train_output["split"] = "train"
            train_output["rsrp_mlp_dbm"] = (
                train_pred
            )

            train_output["signed_error_db"] = (
                train_pred
                -
                y_train
            )

            train_output["abs_error_db"] = np.abs(
                train_output[
                    "signed_error_db"
                ]
            )


            test_output = test_df.copy()

            test_output["split"] = "test"

            test_output["rsrp_mlp_dbm"] = (
                test_pred
            )

            test_output["signed_error_db"] = (
                signed_error
            )

            test_output["abs_error_db"] = (
                abs_error
            )


            predictions = pd.concat(
                [
                    train_output,
                    test_output
                ],
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

                "signed_error_db",
                "abs_error_db"
            ]


            predictions[
                prediction_cols
            ].to_csv(
                OUT / "predictions.csv",
                index=False
            )


            # ====================================================
            # TRAINING HISTORY
            # ====================================================

            history = pd.DataFrame({
                "epoch":
                    np.arange(
                        1,
                        len(model.loss_curve_) + 1
                    ),

                "training_loss":
                    model.loss_curve_
            })

            history.to_csv(
                OUT / "training_history.csv",
                index=False
            )


            # ====================================================
            # SAVE MODEL
            # ====================================================

            joblib.dump(
                model,
                OUT / "model.joblib"
            )

            joblib.dump(
                x_scaler,
                OUT / "x_scaler.joblib"
            )

            joblib.dump(
                y_scaler,
                OUT / "y_scaler.joblib"
            )


            # ====================================================
            # RESOURCE METRICS
            # ====================================================

            if len(training_resources):

                train_cpu_mean = float(
                    training_resources[
                        "process_cpu_percent"
                    ].mean()
                )

                train_cpu_peak = float(
                    training_resources[
                        "process_cpu_percent"
                    ].max()
                )

                train_ram_mean = float(
                    training_resources[
                        "process_ram_mb"
                    ].mean()
                )

                train_ram_peak = float(
                    training_resources[
                        "process_ram_mb"
                    ].max()
                )

            else:

                train_cpu_mean = None
                train_cpu_peak = None
                train_ram_mean = None
                train_ram_peak = None


            if len(inference_resources):

                infer_cpu_mean = float(
                    inference_resources[
                        "process_cpu_percent"
                    ].mean()
                )

                infer_cpu_peak = float(
                    inference_resources[
                        "process_cpu_percent"
                    ].max()
                )

                infer_ram_mean = float(
                    inference_resources[
                        "process_ram_mb"
                    ].mean()
                )

                infer_ram_peak = float(
                    inference_resources[
                        "process_ram_mb"
                    ].max()
                )

            else:

                infer_cpu_mean = None
                infer_cpu_peak = None
                infer_ram_mean = None
                infer_ram_peak = None


            # ====================================================
            # RUN METRICS JSON
            # ====================================================

            metrics = {

                "experiment_id":
                    experiment_id,

                "architecture":
                    arch_name,

                "hidden_layers":
                    list(hidden_layers),

                "seed":
                    seed,

                "features":
                    FEATURES_BASE,

                "train_run":
                    train_run,

                "test_run":
                    test_run,

                "n_train":
                    int(len(train_df)),

                "n_test":
                    int(len(test_df)),

                "height_reference_m":
                    height_reference,

                "epochs":
                    int(model.n_iter_),

                "train_mae_db":
                    float(train_mae),

                "test_mae_db":
                    float(test_mae),

                "generalization_gap_db":
                    float(
                        test_mae
                        -
                        train_mae
                    ),

                "training_runtime_s":
                    float(
                        training_runtime_s
                    ),

                "training_cpu_time_s":
                    float(
                        training_cpu_time_s
                    ),

                "inference_runtime_s":
                    float(
                        inference_runtime_s
                    ),

                "inference_cpu_time_s":
                    float(
                        inference_cpu_time_s
                    ),

                "inference_time_per_sample_us":
                    float(
                        inference_runtime_s
                        /
                        len(test_df)
                        *
                        1e6
                    ),

                "training_cpu_mean_percent":
                    train_cpu_mean,

                "training_cpu_peak_percent":
                    train_cpu_peak,

                "training_ram_mean_mb":
                    train_ram_mean,

                "training_ram_peak_mb":
                    train_ram_peak,

                "inference_cpu_mean_percent":
                    infer_cpu_mean,

                "inference_cpu_peak_percent":
                    infer_cpu_peak,

                "inference_ram_mean_mb":
                    infer_ram_mean,

                "inference_ram_peak_mb":
                    infer_ram_peak,

                "inference_profile_duration_s":
                    INFERENCE_PROFILE_DURATION_S,

                "inference_profile_iterations":
                    profile_iterations
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


            # ====================================================
            # GLOBAL SUMMARY ROW
            # ====================================================

            summary_rows.append(
                metrics
            )


            print(
                f"Epochs: {model.n_iter_:4d} | "
                f"Train MAE: {train_mae:7.3f} dB | "
                f"Test MAE: {test_mae:7.3f} dB | "
                f"Train time: {training_runtime_s:7.3f} s | "
                f"RAM peak: {train_ram_peak}"
            )


# ============================================================
# RAW GLOBAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    ROOT_OUT / "all_runs_summary.csv",
    index=False
)


# ============================================================
# AGGREGATE BY ARCHITECTURE + SPLIT
# ============================================================

group_cols = [
    "architecture",
    "train_run",
    "test_run"
]


aggregate = (
    summary
    .groupby(group_cols)
    .agg(

        n_runs=(
            "test_mae_db",
            "count"
        ),

        test_mae_mean_db=(
            "test_mae_db",
            "mean"
        ),

        test_mae_std_db=(
            "test_mae_db",
            "std"
        ),

        test_mae_min_db=(
            "test_mae_db",
            "min"
        ),

        test_mae_max_db=(
            "test_mae_db",
            "max"
        ),

        train_mae_mean_db=(
            "train_mae_db",
            "mean"
        ),

        generalization_gap_mean_db=(
            "generalization_gap_db",
            "mean"
        ),

        epochs_mean=(
            "epochs",
            "mean"
        ),

        training_runtime_mean_s=(
            "training_runtime_s",
            "mean"
        ),

        training_runtime_std_s=(
            "training_runtime_s",
            "std"
        ),

        training_runtime_total_s=(
            "training_runtime_s",
            "sum"
        ),

        inference_runtime_mean_ms=(
            "inference_runtime_s",
            lambda x:
                x.mean() * 1000
        ),

        inference_time_per_sample_mean_us=(
            "inference_time_per_sample_us",
            "mean"
        ),

        training_ram_peak_mean_mb=(
            "training_ram_peak_mb",
            "mean"
        ),

        training_ram_peak_max_mb=(
            "training_ram_peak_mb",
            "max"
        ),

        inference_ram_peak_mean_mb=(
            "inference_ram_peak_mb",
            "mean"
        ),

        training_cpu_mean_percent=(
            "training_cpu_mean_percent",
            "mean"
        ),

        inference_cpu_mean_percent=(
            "inference_cpu_mean_percent",
            "mean"
        )
    )
    .reset_index()
)


aggregate.to_csv(
    ROOT_OUT / "aggregate_by_architecture_split.csv",
    index=False
)


# ============================================================
# AGGREGATE ACROSS BOTH DIRECTIONS
# ============================================================

overall_arch = (
    summary
    .groupby(
        "architecture"
    )
    .agg(

        n_runs=(
            "test_mae_db",
            "count"
        ),

        test_mae_mean_db=(
            "test_mae_db",
            "mean"
        ),

        test_mae_std_db=(
            "test_mae_db",
            "std"
        ),

        test_mae_min_db=(
            "test_mae_db",
            "min"
        ),

        test_mae_max_db=(
            "test_mae_db",
            "max"
        ),

        train_mae_mean_db=(
            "train_mae_db",
            "mean"
        ),

        generalization_gap_mean_db=(
            "generalization_gap_db",
            "mean"
        ),

        training_runtime_mean_s=(
            "training_runtime_s",
            "mean"
        ),

        training_runtime_total_s=(
            "training_runtime_s",
            "sum"
        ),

        inference_runtime_mean_ms=(
            "inference_runtime_s",
            lambda x:
                x.mean() * 1000
        ),

        training_ram_peak_mean_mb=(
            "training_ram_peak_mb",
            "mean"
        ),

        inference_ram_peak_mean_mb=(
            "inference_ram_peak_mb",
            "mean"
        )
    )
    .reset_index()
)


overall_arch.to_csv(
    ROOT_OUT / "aggregate_overall_architecture.csv",
    index=False
)


# ============================================================
# TOTAL CAMPAIGN COST
# ============================================================

campaign = {

    "n_experiments":
        int(len(summary)),

    "n_seeds":
        len(SEEDS),

    "n_architectures":
        len(ARCHITECTURES),

    "n_split_directions":
        len(SPLITS),

    "total_training_runtime_s":
        float(
            summary[
                "training_runtime_s"
            ].sum()
        ),

    "total_training_cpu_time_s":
        float(
            summary[
                "training_cpu_time_s"
            ].sum()
        ),

    "total_true_inference_runtime_s":
        float(
            summary[
                "inference_runtime_s"
            ].sum()
        ),

    "note":
        (
            "Inference resource profiling repeats inference for "
            "1 s per experiment, but true inference runtime is "
            "stored separately and is not replaced by profiling time."
        )
}


with open(
    ROOT_OUT / "campaign_summary.json",
    "w"
) as f:

    json.dump(
        campaign,
        f,
        indent=2
    )


# ============================================================
# FINAL PRINT
# ============================================================

print()
print("=" * 90)
print("MLP BENCHMARK CAMPAIGN COMPLETE")
print("=" * 90)

print(
    f"Experiments: {len(summary)}"
)

print()

print(
    overall_arch[
        [
            "architecture",
            "n_runs",
            "test_mae_mean_db",
            "test_mae_std_db",
            "training_runtime_mean_s",
            "training_runtime_total_s",
            "training_ram_peak_mean_mb"
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    "Total real training runtime:",
    f"{campaign['total_training_runtime_s']:.3f} s"
)

print()

print("Saved:")
print(ROOT_OUT / "all_runs_summary.csv")
print(ROOT_OUT / "aggregate_by_architecture_split.csv")
print(ROOT_OUT / "aggregate_overall_architecture.csv")
print(ROOT_OUT / "campaign_summary.json")
