import json
import time
import math
import os
import threading
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

DATA = Path("data/processed/powder_benchmark_final.csv")
CONFIG = Path("configs/analytical_3gpp.json")
OUT = Path("results/analytical/run_01")
OUT.mkdir(parents=True, exist_ok=True)

resource_rows = []
stop_monitor = threading.Event()


def monitor_resources(interval=0.02):
    p = psutil.Process(os.getpid())
    t0 = time.perf_counter()
    p.cpu_percent(None)

    while not stop_monitor.is_set():
        resource_rows.append({
            "elapsed_s": time.perf_counter() - t0,
            "cpu_percent": p.cpu_percent(None),
            "ram_mb": p.memory_info().rss / (1024**2),
            "system_ram_percent": psutil.virtual_memory().percent
        })
        time.sleep(interval)


R_EARTH = 6371000.0


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = (
        np.sin(dlat/2)**2
        + np.cos(lat1) * math.cos(lat2) * np.sin(dlon/2)**2
    )

    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def umi_los(d2d, fc_ghz, h_bs, h_ut):
    d2d = np.maximum(d2d, 10.0)

    d3d = np.sqrt(
        d2d**2 + (h_bs-h_ut)**2
    )

    c = 3e8
    fc_hz = fc_ghz * 1e9

    h_bs_eff = h_bs - 1.0
    h_ut_eff = h_ut - 1.0

    d_bp = (
        4 * h_bs_eff * h_ut_eff * fc_hz / c
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
            d_bp**2 + (h_bs-h_ut)**2
        )
    )

    return np.where(d2d <= d_bp, pl1, pl2)


def umi_nlos(d2d, fc_ghz, h_bs, h_ut):
    d2d = np.maximum(d2d, 10.0)

    d3d = np.sqrt(
        d2d**2 + (h_bs-h_ut)**2
    )

    pl_los = umi_los(
        d2d,
        fc_ghz,
        h_bs,
        h_ut
    )

    pl_nlos_prime = (
        35.3*np.log10(d3d)
        + 22.4
        + 21.3*np.log10(fc_ghz)
        - 0.3*(h_ut-1.5)
    )

    return np.maximum(
        pl_los,
        pl_nlos_prime
    )


with open(CONFIG) as f:
    cfg = json.load(f)

df = pd.read_csv(DATA)

df = df.dropna(
    subset=["latitude", "longitude", "rsrp_real_dbm"]
).copy()

d2d = haversine(
    df["latitude"].to_numpy(),
    df["longitude"].to_numpy(),
    cfg["gnb_latitude"],
    cfg["gnb_longitude"]
)

df["distance_2d_m"] = d2d

fc = cfg["frequency_ghz"]
h_bs = cfg["h_bs_m"]
h_ut = cfg["h_ut_m"]

tx_ref = (
    cfg["tx_reference_power_dbm"]
    + cfg["tx_gain_dbi"]
    + cfg["rx_gain_dbi"]
)

# Warm-up
umi_los(d2d[:10], fc, h_bs, h_ut)

thread = threading.Thread(
    target=monitor_resources,
    daemon=True
)
thread.start()

cpu_start = time.process_time()
wall_start = time.perf_counter()

pl_los = umi_los(
    d2d, fc, h_bs, h_ut
)

pl_nlos = umi_nlos(
    d2d, fc, h_bs, h_ut
)

rsrp_los = tx_ref - pl_los
rsrp_nlos = tx_ref - pl_nlos

runtime_s = time.perf_counter() - wall_start
cpu_time_s = time.process_time() - cpu_start

# Profiling loop only for resource trace
profile_start = time.perf_counter()
iterations = 0

while time.perf_counter() - profile_start < 2.0:
    umi_los(d2d, fc, h_bs, h_ut)
    umi_nlos(d2d, fc, h_bs, h_ut)
    iterations += 1

stop_monitor.set()
thread.join()

df["pathloss_los_db"] = pl_los
df["pathloss_nlos_db"] = pl_nlos
df["rsrp_3gpp_los_dbm"] = rsrp_los
df["rsrp_3gpp_nlos_dbm"] = rsrp_nlos

df["abs_error_los_db"] = np.abs(
    df["rsrp_real_dbm"] - rsrp_los
)

df["abs_error_nlos_db"] = np.abs(
    df["rsrp_real_dbm"] - rsrp_nlos
)

mae_los = df["abs_error_los_db"].mean()
mae_nlos = df["abs_error_nlos_db"].mean()

resources = pd.DataFrame(resource_rows)

resources.to_csv(
    OUT / "resources.csv",
    index=False
)

keep = [
    "sample_id",
    "run",
    "timestamp",
    "latitude",
    "longitude",
    "gps_interpolated",
    "rsrp_real_dbm",
    "distance_2d_m",
    "pathloss_los_db",
    "pathloss_nlos_db",
    "rsrp_3gpp_los_dbm",
    "rsrp_3gpp_nlos_dbm",
    "abs_error_los_db",
    "abs_error_nlos_db"
]

df[keep].to_csv(
    OUT / "predictions.csv",
    index=False
)

metrics = {
    "model": cfg["model"],
    "n_samples": int(len(df)),
    "frequency_ghz": fc,
    "mae_los_db": float(mae_los),
    "mae_nlos_db": float(mae_nlos),
    "runtime_single_pass_s": float(runtime_s),
    "time_per_sample_ms": float(runtime_s / len(df) * 1000),
    "cpu_time_single_pass_s": float(cpu_time_s),
    "profile_iterations": int(iterations),
    "cpu_mean_percent": float(resources["cpu_percent"].mean()),
    "cpu_peak_percent": float(resources["cpu_percent"].max()),
    "ram_mean_mb": float(resources["ram_mb"].mean()),
    "ram_peak_mb": float(resources["ram_mb"].max()),
    "max_rss_mb": float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
}

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("="*72)
print("3GPP TR 38.901 ANALYTICAL BASELINE")
print("="*72)

print("Samples:", len(df))
print(
    f"Distance range: {d2d.min():.2f} -> {d2d.max():.2f} m"
)
print(
    f"Observed RSRP range: "
    f"{df['rsrp_real_dbm'].min():.2f} -> "
    f"{df['rsrp_real_dbm'].max():.2f} dBm"
)

print()
print(f"LOS MAE : {mae_los:.3f} dB")
print(f"NLOS MAE: {mae_nlos:.3f} dB")

print()
print(f"Runtime: {runtime_s*1000:.6f} ms")
print(
    f"Time/sample: "
    f"{runtime_s/len(df)*1e6:.6f} us"
)

print()
print(
    f"CPU mean/peak: "
    f"{metrics['cpu_mean_percent']:.2f}% / "
    f"{metrics['cpu_peak_percent']:.2f}%"
)

print(
    f"RAM mean/peak: "
    f"{metrics['ram_mean_mb']:.2f} / "
    f"{metrics['ram_peak_mb']:.2f} MB"
)

print()
print("Saved:")
print(OUT / "predictions.csv")
print(OUT / "resources.csv")
print(OUT / "metrics.json")
