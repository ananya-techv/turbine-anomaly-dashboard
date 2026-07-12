# Turbine Sensor Anomaly Detection Dashboard

A live-simulated monitoring dashboard for a thermal power plant steam
turbine, inspired by the Turbine Supervisory Instrumentation (TSI) and
DCS-based monitoring systems observed during a Control & Instrumentation
internship at a thermal power plant (NTPC/KBUNL).

The dashboard simulates real-time sensor readings — main steam
temperature, main steam pressure, turbine bearing temperature, shaft
vibration, and exhaust vacuum — and flags anomalies (spikes, sustained
drift/faults, and stuck/flatlined sensors) using two interchangeable
detection strategies.

## Why this project

Real plant sensor data isn't publicly available for obvious safety and
confidentiality reasons, so this project simulates realistic sensor
behavior (normal noise, slow drift, and injected fault patterns) to
build and demonstrate a monitoring + anomaly detection pipeline similar
in spirit to what a plant DCS/TSI alarm system does — while keeping the
whole pipeline (data generation, detection logic, dashboard) fully
open and explainable.

## Features

- **Live-streaming simulation** — data is revealed incrementally on a
  timer to mimic a real-time DCS feed, with Play/Pause/Reset controls.
- **Two detection methods**, switchable at runtime:
  - **Rolling Z-Score** (default): flags points that deviate too far
    from a *trailing* rolling mean/std (i.e. only using past readings,
    to avoid a spike inflating the very statistic used to judge it),
    plus a dedicated check for "stuck" sensors (rolling variance
    collapsing to near-zero).
  - **Isolation Forest**: an unsupervised ML model that looks at all
    sensors jointly, useful for catching anomalies that only show up
    as an unusual *combination* of readings.
- **Per-sensor summary cards** with live status (Normal / Anomaly).
- **Overall plant health indicator.**
- **Time-series charts** with anomaly points highlighted and the
  expected normal operating band shaded.
- **Precision / Recall / F1** computed against the simulator's
  ground-truth anomaly labels, shown live in the footer.

## Project structure

```
turbine-anomaly-dashboard/
├── app.py                 # Dash application (UI + callbacks)
├── data_simulator.py       # Generates realistic sensor data + injected faults
├── anomaly_detector.py     # Z-score and Isolation Forest detection logic
├── requirements.txt
├── data/
│   └── turbine_data.csv    # Generated dataset (created on first run)
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python data_simulator.py   # generates data/turbine_data.csv
python app.py               # starts the dashboard
```

Then open **http://127.0.0.1:8050** in your browser. Click **Play** to
start the live simulation, or use the sliders/radio buttons to change
detection method and sensitivity.

## Known limitations & possible improvements

Being upfront about this (and being able to discuss it) is a stronger
signal in an interview than pretending the model is perfect:

- The z-score detector's F1 score on this simulated data is modest
  (~0.15–0.25 depending on sensor and settings). This is expected —
  gradual-onset faults are genuinely hard to catch early with a purely
  statistical threshold, and there's a real precision/recall trade-off
  as the threshold is tuned. A next step would be combining detectors
  (ensemble voting), using a supervised model if labeled fault data
  were available, or adding sensor-specific alarm setpoints like a
  real DCS uses (rate-of-change limits, high-high/low-low bands).
- The simulated data is a simplification — real turbine sensor
  behavior has more coupling between parameters (e.g. steam
  temperature and pressure move together under certain fault
  conditions) than this model captures.
- Isolation Forest is refit on the growing data window each tick for
  simplicity; a production version would train once and score new
  points incrementallly.
  

