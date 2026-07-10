"""
data_simulator.py
------------------
Simulates realistic turbine supervisory instrumentation (TSI) sensor data
for a thermal power plant steam turbine, similar to parameters monitored
in a Turbine Supervisory Instrumentation (TSI) system:

    - Main Steam Temperature (deg C)
    - Main Steam Pressure (kg/cm2)
    - Turbine Bearing Temperature (deg C)
    - Turbine Shaft Vibration (microns)
    - Exhaust (Condenser) Vacuum Pressure (mmHg)

Each parameter is generated around a realistic nominal operating value
with normal process noise, slow drift, and randomly injected anomalies
(spikes, sustained drift-outs, and stuck-sensor flatlines) so the
detection logic has real patterns to catch.

Run this file directly to generate data/turbine_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Nominal operating parameters (typical values referenced from
# thermal power plant turbine operating ranges)
SENSOR_CONFIG = {
    "main_steam_temp_C": {
        "nominal": 535.0, "noise_std": 1.5, "unit": "°C",
        "normal_range": (525, 545), "drift_rate": 0.02
    },
    "main_steam_pressure_kgcm2": {
        "nominal": 150.0, "noise_std": 0.8, "unit": "kg/cm²",
        "normal_range": (145, 155), "drift_rate": 0.015
    },
    "bearing_temp_C": {
        "nominal": 78.0, "noise_std": 1.0, "unit": "°C",
        "normal_range": (65, 90), "drift_rate": 0.03
    },
    "shaft_vibration_microns": {
        "nominal": 35.0, "noise_std": 2.5, "unit": "µm",
        "normal_range": (0, 75), "drift_rate": 0.04
    },
    "exhaust_vacuum_mmHg": {
        "nominal": 685.0, "noise_std": 1.2, "unit": "mmHg",
        "normal_range": (660, 700), "drift_rate": 0.02
    },
}


def _generate_clean_series(n_points, nominal, noise_std, drift_rate, seed):
    rng = np.random.default_rng(seed)
    # slow random-walk drift so the signal doesn't look purely random
    drift = np.cumsum(rng.normal(0, drift_rate, n_points))
    noise = rng.normal(0, noise_std, n_points)
    series = nominal + drift + noise
    return series


def _inject_anomalies(series, rng, anomaly_rate=0.02):
    """
    Injects three anomaly types into a copy of the series:
      1. Spike       - single point jumps far out of range, then recovers
      2. Sustained    - a block of points drifts/stays out of range
      3. Flatline     - sensor 'sticks' at a constant value (stuck sensor)

    Returns (series_with_anomalies, anomaly_labels) where labels is a
    boolean array marking ground-truth anomalous points.
    """
    n = len(series)
    series = series.copy()
    labels = np.zeros(n, dtype=bool)

    n_events = max(1, int(n * anomaly_rate / 8))  # roughly space out events

    for _ in range(n_events):
        event_type = rng.choice(["spike", "sustained", "flatline"], p=[0.5, 0.3, 0.2])
        start = rng.integers(50, n - 50)

        if event_type == "spike":
            width = rng.integers(1, 3)
            magnitude = rng.choice([-1, 1]) * rng.uniform(4, 8) * series.std()
            end = min(start + width, n)
            series[start:end] += magnitude
            labels[start:end] = True

        elif event_type == "sustained":
            # short ramp-in (e.g. a valve/fault developing quickly) then holds
            # at an out-of-range level -- more realistic than a slow linear
            # ramp across the whole event, and easier to detect close to
            # onset, same as a real DCS alarm would catch it
            width = rng.integers(15, 40)
            ramp_in = min(5, width)
            magnitude = rng.choice([-1, 1]) * rng.uniform(3, 6) * series.std()
            end = min(start + width, n)
            actual_width = end - start
            ramp = np.concatenate([
                np.linspace(0, magnitude, ramp_in),
                np.full(max(actual_width - ramp_in, 0), magnitude),
            ])[:actual_width]
            series[start:end] += ramp
            labels[start:end] = True

        elif event_type == "flatline":
            width = rng.integers(10, 25)
            end = min(start + width, n)
            stuck_value = series[start]
            series[start:end] = stuck_value
            labels[start:end] = True

    return series, labels


def generate_turbine_data(n_points=2000, interval_seconds=30, anomaly_rate=0.03,
                           start_time=None, seed=42):
    """
    Generates a full simulated turbine sensor dataset.

    Parameters
    ----------
    n_points : int
        Number of timestamped readings to generate per sensor.
    interval_seconds : int
        Time gap between readings (simulates a real DCS scan rate).
    anomaly_rate : float
        Approximate fraction of points involved in anomalous events.
    start_time : datetime, optional
        Timestamp of the first reading. Defaults to n_points*interval_seconds
        before now.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with a 'timestamp' column, one column per sensor, and
    one '<sensor>_is_anomaly' ground-truth label column per sensor.
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(seconds=n_points * interval_seconds)

    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(n_points)]
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"timestamp": timestamps})

    for i, (sensor_name, cfg) in enumerate(SENSOR_CONFIG.items()):
        clean = _generate_clean_series(
            n_points, cfg["nominal"], cfg["noise_std"], cfg["drift_rate"], seed=seed + i
        )
        with_anomalies, labels = _inject_anomalies(clean, rng, anomaly_rate=anomaly_rate)
        df[sensor_name] = with_anomalies
        df[f"{sensor_name}_is_anomaly"] = labels

    return df


if __name__ == "__main__":
    df = generate_turbine_data(n_points=2000, interval_seconds=30, anomaly_rate=0.03)
    df.to_csv("data/turbine_data.csv", index=False)
    total_anomalies = sum(df[c].sum() for c in df.columns if c.endswith("_is_anomaly"))
    print(f"Generated {len(df)} readings -> data/turbine_data.csv")
    print(f"Total ground-truth anomalous points across all sensors: {total_anomalies}")
