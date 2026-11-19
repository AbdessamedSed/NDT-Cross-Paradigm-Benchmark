# NDT Cross-Paradigm Benchmark

Reproducibility repository for an empirical comparison of modeling paradigms
for 5G Network Digital Twins (NDTs).

The benchmark compares four modeling approaches against real-world measurements
collected on the POWDER wireless research platform:

- **Analytical:** 3GPP TR 38.901 UMi
- **Simulation-based:** OMNeT++ / Simu5G
- **AI-based:** Multi-Layer Perceptron (MLP)
- **Hybrid:** Sionna RT + ns-3 / 5G-LENA

## Research objective

The benchmark investigates how different NDT modeling paradigms trade
physical-network fidelity against computational cost, data requirements,
and site-specific information.

All approaches are evaluated against the same:

- physical deployment;
- UE trajectories;
- radio configuration;
- target quantity;
- POWDER measurements;
- cross-run evaluation protocol.

The target is the serving-cell NR RSRP:

```text
RSRP (NR SpCell)
