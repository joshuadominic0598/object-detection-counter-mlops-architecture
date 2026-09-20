"""
Shared config, Mongo access, and HTML helpers used by both
generate_monitoring_dashboard.py and generate_session_dashboard.py.
"""

import base64
import html
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


# Config
load_dotenv()

OUTPUT_DIR = "tmp/dashboard"

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB = os.getenv("MONGO_DB", "prod_counter")

REQUESTS_PER_PAGE = 20

# model_management/evaluation/scripts/run_performance_tests.py tags its sessions with this prefix -
# that traffic is isolated performance testing, not real monitoring data.
PERFORMANCE_SESSION_PREFIX = "perf-test-"

LEGACY_ENDPOINTS = {
    "/object-count",
    "/object-list",
}


# Mongo connection
client = MongoClient(MONGO_HOST, MONGO_PORT)
database = client[MONGO_DB]

detection_history_col = database.detection_history
counter_col = database.counter


# Chart theme
CHART_FONT_COLOR = "#cbd5e1"
CHART_TITLE_COLOR = "#f8fafc"
CHART_GRID_COLOR = "#1f2937"


def apply_chart_theme(fig):
    """Apply the dashboard's dark theme to a Plotly figure."""

    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color=CHART_FONT_COLOR),
        title_font=dict(color=CHART_TITLE_COLOR),
        legend=dict(font=dict(color=CHART_FONT_COLOR)),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor=CHART_GRID_COLOR,
            font_color=CHART_TITLE_COLOR,
        ),
    )

    fig.update_xaxes(
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
        linecolor=CHART_GRID_COLOR,
    )

    fig.update_yaxes(
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
        linecolor=CHART_GRID_COLOR,
    )

    return fig


# Text and value helpers

def image_to_base64(image_path):
    """Convert a local result image into a data URI so the dashboard stays self-contained."""

    if not image_path:
        return None

    path = Path(str(image_path))

    if not path.exists():
        return None

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return "data:image/jpeg;base64," + encoded
    except OSError:
        return None


def safe_text(value):
    """Escape values before inserting them into HTML."""

    if value is None:
        return ""

    return html.escape(str(value))


def format_timestamp(value):
    """Format timestamps consistently."""

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"

    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_counts(counts):
    """Convert a counts dictionary into HTML."""

    if not isinstance(counts, dict) or not counts:
        return '<span class="muted">No detections</span>'

    rows = []

    for object_class, count_value in sorted(counts.items(), key=lambda item: str(item[0])):
        try:
            count_value = int(count_value)
        except (ValueError, TypeError):
            continue

        rows.append(
            f"""
            <div class="count-row">
                <span>{safe_text(object_class)}</span>
                <strong>{count_value:,}</strong>
            </div>
            """
        )

    if not rows:
        return '<span class="muted">No detections</span>'

    return "".join(rows)


def infer_counter_flag(row):
    """
    Resolve whether a request used counting.

    Older records predate the `counter` field, so fall back
    to the legacy endpoint name to infer the same meaning.
    """

    counter_value = row.get("counter")

    if pd.notna(counter_value):
        return bool(counter_value)

    endpoint = row.get("endpoint")

    if endpoint == "/object-count":
        return True

    if endpoint == "/object-list":
        return False

    return None


def normalize_endpoint(endpoint):
    """Map legacy request endpoints to the current single API route."""

    if endpoint in LEGACY_ENDPOINTS:
        return "/object-detection"

    return endpoint


# Request cards (used by the session dashboard)

def build_request_card(request):
    """Build one request card."""

    image_uri = image_to_base64(request.get("result_image_path"))
    success = bool(request.get("success", False))
    image_name = safe_text(request.get("image_name", "image"))

    if image_uri:
        image_html = f"""
        <div class="result-image-container">
            <img class="result-image" src="{image_uri}" alt="Annotated result for {image_name}">
        </div>
        """
    else:
        image_message = "Result image unavailable" if success else "No prediction image available"
        image_html = f"""
        <div class="result-image-unavailable">
            {safe_text(image_message)}
        </div>
        """

    status_class = "status-success" if success else "status-failed"
    status_text = "Successful" if success else "Failed"

    response_time = request.get("response_time_ms")

    if response_time is None or pd.isna(response_time):
        response_text = "N/A"
    else:
        response_text = f"{float(response_time):.0f} ms"

    threshold = request.get("threshold")

    if threshold is None or pd.isna(threshold):
        threshold_text = "N/A"
    else:
        threshold_text = f"{float(threshold):.2f}"

    detected_count = request.get("detected_count", 0)

    try:
        detected_count = int(detected_count)
    except (ValueError, TypeError):
        detected_count = 0

    error_message = request.get("error_message")
    error_html = ""

    if error_message:
        error_html = f"""
        <div class="error-message">
            <strong>Error:</strong> {safe_text(error_message)}
        </div>
        """

    # counter: True -> counting call (persisted counts snapshot available)
    #          False -> object-list call (no persistence happens)
    #          None -> legacy record, unknown call type
    counter_value = request.get("counter")

    if counter_value is True:
        counter_text = "True"
    elif counter_value is False:
        counter_text = "False"
    else:
        counter_text = "N/A"

    if counter_value is True:
        persisted_section = f"""
        <div class="request-counts">
            <div class="counts-title">Persisted Counts (at this point)</div>
            {format_counts(request.get("persisted_counts"))}
        </div>
        """
    elif counter_value is False:
        persisted_section = """
        <div class="request-counts">
            <div class="counts-title">Persisted Counts</div>
            <span class="muted">Not tracked for object-list calls</span>
        </div>
        """
    else:
        persisted_section = ""

    return f"""
    <div class="request-card">
        <div class="request-card-header">
            <div>
                <div class="request-image-name">{image_name}</div>
                <div class="request-timestamp">{format_timestamp(request.get("request_timestamp"))}</div>
            </div>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>

        {image_html}

        <div class="request-details">
            <div class="detail-item">
                <span>Endpoint</span>
                <strong>{safe_text(request.get("endpoint", "unknown"))}</strong>
            </div>
            <div class="detail-item">
                <span>Threshold</span>
                <strong>{threshold_text}</strong>
            </div>
            <div class="detail-item">
                <span>Total Detections</span>
                <strong>{detected_count:,}</strong>
            </div>
            <div class="detail-item">
                <span>Response Time</span>
                <strong>{response_text}</strong>
            </div>
            <div class="detail-item">
                <span>Counter</span>
                <strong>{counter_text}</strong>
            </div>
        </div>

        <div class="request-counts">
            <div class="counts-title">Counts</div>
            {format_counts(request.get("counts"))}
        </div>

        {persisted_section}
        {error_html}
    </div>
    """


def build_session_request_pages(session_df, session_index):
    """
    Build paginated request cards.

    Requests are ordered newest first. Each page contains at most
    REQUESTS_PER_PAGE requests.
    """

    session_df = session_df.sort_values("request_timestamp", ascending=False).reset_index(drop=True)

    if session_df.empty:
        return """
        <div class="empty-state">
            No request history is available for this session.
        </div>
        """

    total_requests = len(session_df)
    total_pages = (total_requests + REQUESTS_PER_PAGE - 1) // REQUESTS_PER_PAGE

    pages = []

    for page_index in range(total_pages):
        start = page_index * REQUESTS_PER_PAGE
        end = min(start + REQUESTS_PER_PAGE, total_requests)
        page_df = session_df.iloc[start:end]

        cards = [build_request_card(request) for _, request in page_df.iterrows()]

        page_id = f"session-{session_index}-requests-page-{page_index}"
        display_style = "block" if page_index == 0 else "none"

        pages.append(
            f"""
            <div id="{page_id}" class="request-page" style="display: {display_style};">
                <div class="request-card-grid">
                    {"".join(cards)}
                </div>
            </div>
            """
        )

    pagination_id = f"session-{session_index}-request-pagination"

    pagination_html = f"""
    <div id="{pagination_id}" class="request-pagination">
        <button class="pagination-button" onclick="changeRequestPage('{pagination_id}', -1)" type="button">
            ← Previous
        </button>
        <span class="pagination-status" data-current-page="1" data-total-pages="{total_pages}">
            1 / {total_pages}
        </span>
        <button class="pagination-button" onclick="changeRequestPage('{pagination_id}', 1)" type="button">
            Next →
        </button>
    </div>
    """

    return f"""
    <div class="request-pages-container" data-total-pages="{total_pages}">
        {"".join(pages)}
    </div>

    {pagination_html}
    """


# Data loading

def load_requests_df():
    """Load and clean the detection history collection into a DataFrame."""

    documents = list(
        detection_history_col.find(
            {},
            {
                "_id": 0,
                "request_id": 1,
                "session_id": 1,
                "request_timestamp": 1,
                "endpoint": 1,
                "image_name": 1,
                "threshold": 1,
                "response_time_ms": 1,
                "detected_count": 1,
                "detected_classes": 1,
                "counts": 1,
                "status_code": 1,
                "success": 1,
                "error_message": 1,
                "result_image_path": 1,
                "counter": 1,
                "persisted_counts": 1,
            },
        )
    )

    requests_df = pd.DataFrame(documents)

    if requests_df.empty:
        return requests_df

    requests_df["request_timestamp"] = pd.to_datetime(
        requests_df["request_timestamp"], utc=True, errors="coerce"
    )

    requests_df["success"] = requests_df["success"].fillna(False).astype(bool)
    requests_df["session_id"] = requests_df["session_id"].fillna("unknown").astype(str)
    requests_df["endpoint"] = requests_df["endpoint"].fillna("unknown").astype(str)

    requests_df["counter"] = requests_df.apply(infer_counter_flag, axis=1)
    requests_df["endpoint"] = requests_df["endpoint"].map(normalize_endpoint)

    requests_df = requests_df[
        ~requests_df["session_id"].str.startswith(PERFORMANCE_SESSION_PREFIX)
    ].reset_index(drop=True)

    if requests_df.empty:
        return requests_df

    requests_df["detected_count"] = pd.to_numeric(
        requests_df["detected_count"], errors="coerce"
    ).fillna(0)

    requests_df["response_time_ms"] = pd.to_numeric(requests_df["response_time_ms"], errors="coerce")
    requests_df["threshold"] = pd.to_numeric(requests_df["threshold"], errors="coerce")

    if "counter" not in requests_df.columns:
        requests_df["counter"] = None

    if "persisted_counts" not in requests_df.columns:
        requests_df["persisted_counts"] = None

    return requests_df


def load_session_counters():
    """Load persisted cumulative counts per session."""

    counter_documents = list(
        counter_col.find(
            {},
            {"_id": 0, "session_id": 1, "object_class": 1, "count": 1},
        )
    )

    session_counters = {}

    for document in counter_documents:
        session_id = str(document.get("session_id", "unknown"))
        object_class = document.get("object_class")
        count_value = document.get("count", 0)

        try:
            count_value = int(count_value)
        except (ValueError, TypeError):
            count_value = 0

        session_counters.setdefault(session_id, {})[object_class] = count_value

    return session_counters


def build_session_summary(requests_df):
    """Aggregate per-session metrics, newest session first."""

    session_summary = (
        requests_df.groupby("session_id")
        .agg(
            requests=("request_id", "count"),
            detections=("detected_count", "sum"),
            avg_response_ms=("response_time_ms", "mean"),
            success_rate=("success", "mean"),
            first_request=("request_timestamp", "min"),
            last_request=("request_timestamp", "max"),
        )
        .reset_index()
    )

    session_summary["avg_response_ms"] = session_summary["avg_response_ms"].round(2)
    session_summary["success_rate"] = (session_summary["success_rate"] * 100).round(2)
    session_summary["detections"] = session_summary["detections"].astype(int)

    return session_summary.sort_values("last_request", ascending=False)


# Output

def write_dashboard(filename, html_output):
    """Write a generated dashboard to OUTPUT_DIR and report the path."""

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = Path(OUTPUT_DIR) / filename
    output_file.write_text(html_output, encoding="utf-8")
    print(f"Dashboard generated: {output_file}")
    return output_file


def empty_dashboard_html(title, message):
    """Render a minimal placeholder page when there is no data yet."""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{safe_text(title)}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #000;
                color: #f8fafc;
                padding: 40px;
            }}
            .message {{
                max-width: 700px;
                margin: 80px auto;
                background: #05070d;
                padding: 40px;
                border: 1px solid #1f2937;
                border-radius: 12px;
                text-align: center;
                box-shadow:
                    0 0 0 3px rgba(37, 99, 235, 0.26),
                    0 18px 45px rgba(0, 0, 0, 0.6);
            }}
            p {{
                color: #9ca3af;
            }}
        </style>
    </head>
    <body>
        <div class="message">
            <h1>{safe_text(title)}</h1>
            <p>{safe_text(message)}</p>
        </div>
    </body>
    </html>
    """
