# metrics.py
import os
import time

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
)

SERVICE_NAME = os.getenv("SERVICE_NAME", "chord-bot-local")

#tot requests by status
REQUEST_COUNT = Counter(
    "chordbot_requests_total",
    "Total number of Chord-Bot requests",
    ["service", "endpoint", "status"],  # status: success | invalid_input | error
)

#latency
REQUEST_LATENCY = Histogram(
    "chordbot_request_latency_seconds",
    "Latency of Chord-Bot requests in seconds",
    ["service", "endpoint"],
)

# Number of note tokens per request
NOTES_PER_REQUEST = Histogram(
    "chordbot_notes_per_request",
    "Number of note tokens received per request",
    ["service", "endpoint"],
    
    buckets=(1, 2, 3, 4, 5, 6, 8, 12, 16),
)

INVALID_REQUESTS = Counter(
    "chordbot_invalid_requests_total",
    "Number of Chord-Bot requests with invalid input",
    ["service", "endpoint"],
)

#counts of predicted chords by label
CHORD_PREDICTIONS = Counter(
    "chordbot_chord_predictions_total",
    "Number of times each chord label is predicted",
    ["service", "endpoint", "chord_label"],
)

#how many requests
ACTIVE_REQUESTS = Gauge(
    "chordbot_active_requests",
    "Number of active Chord-Bot requests being processed",
    ["service", "endpoint"],
)


def init_metrics(port: int = 8000, service_name: str | None = None) -> None:
    """
    Start the Prometheus metrics HTTP server on `port`.

    Call this once at app startup (before serving requests).
    """
    global SERVICE_NAME

    if service_name is not None:
        SERVICE_NAME = service_name

    #expose on http://0.0.0.0:port/metrics
    start_http_server(port)
    print(f"[metrics] Prometheus metrics server started on :{port} for service={SERVICE_NAME}")


def instrument_chord_request(endpoint: str, note_tokens, chord_label: str | None, status: str, elapsed_seconds: float) -> None:
    """
    Helper you can call from your endpoint / handler *after* processing a request.

    Parameters:
        endpoint       - logical endpoint name, e.g. "analyze" or "predict"
        note_tokens    - list of parsed notes (e.g., ["C", "E", "G"]) or None
        chord_label    - predicted chord label (e.g., "Cmaj") or None if invalid/failed
        status         - "success", "invalid_input", or "error"
        elapsed_seconds - wall-clock latency for the request
    """

    REQUEST_COUNT.labels(
        service=SERVICE_NAME,
        endpoint=endpoint,
        status=status,
    ).inc()

    REQUEST_LATENCY.labels(
        service=SERVICE_NAME,
        endpoint=endpoint,
    ).observe(elapsed_seconds)

    if note_tokens is not None:
        NOTES_PER_REQUEST.labels(
            service=SERVICE_NAME,
            endpoint=endpoint,
        ).observe(len(note_tokens))

    if status == "invalid_input":
        INVALID_REQUESTS.labels(
            service=SERVICE_NAME,
            endpoint=endpoint,
        ).inc()

    if status == "success" and chord_label is not None:
        CHORD_PREDICTIONS.labels(
            service=SERVICE_NAME,
            endpoint=endpoint,
            chord_label=chord_label,
        ).inc()
