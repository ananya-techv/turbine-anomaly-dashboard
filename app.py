"""
app.py
------
Turbine Sensor Anomaly Detection Dashboard

A Dash web app that simulates a live feed of turbine supervisory
instrumentation (TSI) data -- main steam temperature/pressure, bearing
temperature, shaft vibration, and exhaust vacuum -- and flags anomalies
in real time using a rolling z-score detector (with an Isolation Forest
option for multivariate detection).

Run with:  python app.py
Then open: http://127.0.0.1:8050
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

from data_simulator import generate_turbine_data, SENSOR_CONFIG
from anomaly_detector import detect_anomalies_zscore, detect_anomalies_isolation_forest, evaluate_detection

# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "turbine_data.csv")

if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    df_full = generate_turbine_data(n_points=2000, interval_seconds=30, anomaly_rate=0.03)
    df_full.to_csv(DATA_PATH, index=False)
else:
    df_full = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

SENSOR_NAMES = list(SENSOR_CONFIG.keys())
LABELS = {
    "main_steam_temp_C": "Main Steam Temperature",
    "main_steam_pressure_kgcm2": "Main Steam Pressure",
    "bearing_temp_C": "Turbine Bearing Temperature",
    "shaft_vibration_microns": "Shaft Vibration",
    "exhaust_vacuum_mmHg": "Exhaust Vacuum",
}

STREAM_START = 100          # start the "live" view after this many points, so rolling stats have history
STREAM_STEP = 5              # how many new points revealed per tick
TOTAL_POINTS = len(df_full)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Turbine Anomaly Dashboard"

COLORS = {
    "bg": "#0f1720",
    "panel": "#182432",
    "text": "#e6edf3",
    "muted": "#8b98a5",
    "accent": "#3fb1ce",
    "ok": "#2ecc71",
    "warn": "#f5a623",
    "bad": "#e5484d",
}

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": "Inter, Segoe UI, sans-serif",
           "padding": "24px", "color": COLORS["text"]},
    children=[
        html.Div([
            html.H2("Turbine Sensor Anomaly Detection Dashboard", style={"marginBottom": "2px"}),
            html.P("Simulated Turbine Supervisory Instrumentation (TSI) feed — live anomaly monitoring",
                   style={"color": COLORS["muted"], "marginTop": "0"}),
        ]),

        # Controls
        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "center", "flexWrap": "wrap",
                   "backgroundColor": COLORS["panel"], "padding": "14px 18px", "borderRadius": "10px",
                   "marginBottom": "18px"},
            children=[
                html.Div([
                    html.Label("Detection method", style={"color": COLORS["muted"], "fontSize": "12px"}),
                    dcc.RadioItems(
                        id="method-select",
                        options=[
                            {"label": " Rolling Z-Score", "value": "zscore"},
                            {"label": " Isolation Forest", "value": "iforest"},
                        ],
                        value="zscore",
                        labelStyle={"marginRight": "14px", "color": COLORS["text"]},
                        inputStyle={"marginRight": "6px"},
                    ),
                ]),
                html.Div([
                    html.Label("Z-score threshold", style={"color": COLORS["muted"], "fontSize": "12px"}),
                    dcc.Slider(id="zscore-threshold", min=1.5, max=5, step=0.5, value=2.0,
                               marks={i: str(i) for i in [2, 3, 4, 5]}),
                ], style={"width": "220px"}),
                html.Div([
                    html.Label("Live playback", style={"color": COLORS["muted"], "fontSize": "12px"}),
                    html.Div([
                        html.Button("Play", id="play-btn", n_clicks=0,
                                    style={"marginRight": "8px", "padding": "6px 14px", "borderRadius": "6px",
                                           "border": "none", "backgroundColor": COLORS["accent"], "color": "#fff",
                                           "cursor": "pointer"}),
                        html.Button("Reset", id="reset-btn", n_clicks=0,
                                    style={"padding": "6px 14px", "borderRadius": "6px", "border": "none",
                                           "backgroundColor": "#333", "color": "#fff", "cursor": "pointer"}),
                    ]),
                ]),
            ],
        ),

        # Plant health + summary cards
        html.Div(id="summary-cards", style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                                             "marginBottom": "18px"}),

        # Charts
        html.Div(id="sensor-charts"),

        # Hidden state
        dcc.Interval(id="tick", interval=800, n_intervals=0, disabled=True),
        dcc.Store(id="cursor", data=STREAM_START),
        dcc.Store(id="is-playing", data=False),

        html.Div(id="perf-footer", style={"color": COLORS["muted"], "fontSize": "12px", "marginTop": "10px"}),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("is-playing", "data"),
    Output("tick", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, is_playing):
    new_state = not is_playing
    return new_state, (not new_state), ("Pause" if new_state else "Play")


@app.callback(
    Output("cursor", "data"),
    Input("tick", "n_intervals"),
    Input("reset-btn", "n_clicks"),
    State("cursor", "data"),
    prevent_initial_call=True,
)
def advance_cursor(n_intervals, reset_clicks, cursor):
    from dash import ctx
    if ctx.triggered_id == "reset-btn":
        return STREAM_START
    new_cursor = cursor + STREAM_STEP
    if new_cursor >= TOTAL_POINTS:
        new_cursor = STREAM_START
    return new_cursor


def _status_color(is_anom_now):
    return COLORS["bad"] if is_anom_now else COLORS["ok"]


@app.callback(
    Output("summary-cards", "children"),
    Output("sensor-charts", "children"),
    Output("perf-footer", "children"),
    Input("cursor", "data"),
    Input("method-select", "value"),
    Input("zscore-threshold", "value"),
)
def update_dashboard(cursor, method, zscore_threshold):
    window_df = df_full.iloc[:cursor].copy()

    cards = []
    charts = []
    any_anomaly_now = False
    f1_scores = []

    if method == "iforest":
        iforest_flags = detect_anomalies_isolation_forest(window_df, SENSOR_NAMES)

    for sensor in SENSOR_NAMES:
        cfg = SENSOR_CONFIG[sensor]
        series = window_df[sensor]
        true_labels = window_df[f"{sensor}_is_anomaly"].astype(bool)

        if method == "zscore":
            is_anom, z = detect_anomalies_zscore(series, window=15, threshold=zscore_threshold)
        else:
            is_anom = iforest_flags
            z = None

        perf = evaluate_detection(true_labels, is_anom)
        f1_scores.append(perf["f1"])

        latest_val = series.iloc[-1] if len(series) else cfg["nominal"]
        latest_anom = bool(is_anom.iloc[-1]) if len(is_anom) else False
        any_anomaly_now = any_anomaly_now or latest_anom

        # --- summary card ---
        cards.append(html.Div(
            style={"backgroundColor": COLORS["panel"], "borderRadius": "10px", "padding": "14px 16px",
                   "minWidth": "190px", "borderLeft": f"4px solid {_status_color(latest_anom)}"},
            children=[
                html.Div(LABELS[sensor], style={"color": COLORS["muted"], "fontSize": "12px"}),
                html.Div(f"{latest_val:,.1f} {cfg['unit']}", style={"fontSize": "22px", "fontWeight": "600"}),
                html.Div("ANOMALY" if latest_anom else "Normal",
                         style={"color": _status_color(latest_anom), "fontSize": "12px", "fontWeight": "600"}),
            ],
        ))

        # --- time series chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=window_df["timestamp"], y=series, mode="lines",
            line=dict(color=COLORS["accent"], width=1.5), name=LABELS[sensor],
        ))
        anom_points = window_df.loc[is_anom.values, "timestamp"] if len(is_anom) else []
        anom_vals = series.loc[is_anom.values] if len(is_anom) else []
        fig.add_trace(go.Scatter(
            x=anom_points, y=anom_vals, mode="markers",
            marker=dict(color=COLORS["bad"], size=7, symbol="circle-open", line=dict(width=2)),
            name="Flagged anomaly",
        ))
        low, high = cfg["normal_range"]
        fig.add_hrect(y0=low, y1=high, fillcolor=COLORS["ok"], opacity=0.06, line_width=0)

        fig.update_layout(
            title=f"{LABELS[sensor]}  ({cfg['unit']})  —  F1: {perf['f1']}",
            template="plotly_dark",
            paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
            margin=dict(l=40, r=20, t=40, b=30), height=260,
            showlegend=False,
        )
        charts.append(dcc.Graph(figure=fig, style={"marginBottom": "12px"}))

    plant_status = html.Div(
        style={"backgroundColor": COLORS["panel"], "borderRadius": "10px", "padding": "14px 16px",
               "minWidth": "190px", "borderLeft": f"4px solid {_status_color(any_anomaly_now)}"},
        children=[
            html.Div("PLANT STATUS", style={"color": COLORS["muted"], "fontSize": "12px"}),
            html.Div("ANOMALY DETECTED" if any_anomaly_now else "All Systems Normal",
                     style={"fontSize": "18px", "fontWeight": "700", "color": _status_color(any_anomaly_now)}),
            html.Div(f"{cursor}/{TOTAL_POINTS} readings streamed", style={"fontSize": "11px", "color": COLORS["muted"]}),
        ],
    )
    cards = [plant_status] + cards

    avg_f1 = round(float(np.mean(f1_scores)), 3) if f1_scores else 0.0
    footer = f"Detection method: {'Rolling Z-Score' if method == 'zscore' else 'Isolation Forest'}  |  Avg. F1 vs ground-truth simulated anomalies: {avg_f1}"

    return cards, charts, footer


if __name__ == "__main__":
    app.run(debug=True, port=8050)
