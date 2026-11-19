import math
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "ndt_modeling_benchmark"
INPUT = ROOT / "data/processed/powder_benchmark_final.csv"
OUT = ROOT / "models/hybrid_sionna_lena"
OUT.mkdir(parents=True, exist_ok=True)

# Nominal MEB gNB coordinate used by the benchmark
LAT0 = 40.76855
LON0 = -111.84616

R_EARTH = 6371000.0  # m

df = pd.read_csv(INPUT)

def latlon_to_local(lat, lon):
    lat0 = math.radians(LAT0)
    dlat = math.radians(lat - LAT0)
    dlon = math.radians(lon - LON0)

    # Local tangent-plane approximation
    x = R_EARTH * dlon * math.cos(lat0)   # East
    y = R_EARTH * dlat                    # North

    return x, y

for run in ["run1", "run2"]:
    d = df[df["run"] == run].copy().reset_index(drop=True)

    # Relative simulation time
    d["sim_time_s"] = d["time_s"] - d["time_s"].iloc[0]

    xy = d.apply(
        lambda r: latlon_to_local(r["latitude"], r["longitude"]),
        axis=1
    )

    d["x_m"] = [p[0] for p in xy]
    d["y_m"] = [p[1] for p in xy]

    # IMPORTANT: local radio height, not GNSS altitude
    d["z_m"] = 1.5

    out = d[
        [
            "sample_id",
            "sim_time_s",
            "x_m",
            "y_m",
            "z_m",
            "latitude",
            "longitude",
            "rsrp_real_dbm",
            "gps_interpolated",
        ]
    ]

    path = OUT / f"powder_{run}_trajectory.csv"
    out.to_csv(path, index=False)

    print(f"\n=== {run.upper()} ===")
    print("Samples:", len(out))
    print("Duration:", out["sim_time_s"].iloc[-1], "s")
    print("X range:", out["x_m"].min(), "->", out["x_m"].max(), "m")
    print("Y range:", out["y_m"].min(), "->", out["y_m"].max(), "m")
    print("RSRP range:", out["rsrp_real_dbm"].min(),
          "->", out["rsrp_real_dbm"].max(), "dBm")
    print("Saved:", path)

print("\nNominal gNB local coordinates:")
print("x = 0 m")
print("y = 0 m")
print("z = 10 m")
