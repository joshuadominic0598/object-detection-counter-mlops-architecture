"""
Generates the session dashboard: the latest 5 sessions with persisted
counts, API call distribution, and full paginated request history
(including annotated result images) for debugging individual sessions.

For overall/operational metrics, see generate_monitoring_dashboard.py.
"""

import pandas as pd

from helpers import (
    build_session_request_pages,
    build_session_summary,
    empty_dashboard_html,
    format_counts,
    format_timestamp,
    load_requests_df,
    load_session_counters,
    safe_text,
    write_dashboard,
)


OUTPUT_FILENAME = "session_dashboard.html"

LATEST_SESSION_COUNT = 5


def build_session_sections(requests_df, session_summary, session_counters):
    latest_sessions = session_summary.head(LATEST_SESSION_COUNT)["session_id"].tolist()

    session_cards = []
    session_detail_sections = []

    for index, session_id in enumerate(latest_sessions):
        session_df = requests_df[requests_df["session_id"] == session_id].copy()
        session_row = session_summary[session_summary["session_id"] == session_id].iloc[0]

        requests_count = int(session_row["requests"])
        detections_count = int(session_row["detections"])
        avg_response = session_row["avg_response_ms"]
        success_rate = session_row["success_rate"]

        avg_response_text = "N/A" if pd.isna(avg_response) else f"{avg_response:.2f} ms"
        success_text = "N/A" if pd.isna(success_rate) else f"{success_rate:.2f}%"

        cumulative_counts_html = format_counts(session_counters.get(session_id, {}))

        api_call_distribution = session_df["endpoint"].value_counts().to_dict()
        api_call_distribution_html = format_counts(api_call_distribution)

        detail_id = f"session-detail-{index}"

        session_cards.append(
            f"""
            <button
                class="session-card{' active' if index == 0 else ''}"
                data-session-target="{detail_id}"
                onclick="showSession('{detail_id}')"
                type="button"
            >
                <div class="session-card-title">{safe_text(session_id)}</div>
                <div class="session-card-metrics">
                    <div><span>Requests</span><strong>{requests_count:,}</strong></div>
                    <div><span>Detections</span><strong>{detections_count:,}</strong></div>
                    <div><span>Success</span><strong>{success_text}</strong></div>
                </div>
                <div class="session-card-counts">
                    <span>Persisted Counts</span>
                    <div class="cumulative-counts">{cumulative_counts_html}</div>
                </div>
                <div class="session-card-footer">Click to view session details</div>
            </button>
            """
        )

        request_pages_html = build_session_request_pages(session_df, index)

        session_detail_sections.append(
            f"""
            <div id="{detail_id}" class="session-detail {'active' if index == 0 else ''}">
                <button class="session-detail-toggle" onclick="closeSession('{detail_id}')" type="button">
                    <div class="session-detail-toggle-content">
                        <div class="session-detail-label">Selected Session</div>
                        <h2>{safe_text(session_id)}</h2>
                    </div>
                </button>

                <div class="session-detail-content" id="{detail_id}-content">
                    <div class="session-metric-grid">
                        <div class="session-metric-card">
                            <span>Requests</span>
                            <strong>{requests_count:,}</strong>
                        </div>
                        <div class="session-metric-card">
                            <span>Detections</span>
                            <strong>{detections_count:,}</strong>
                        </div>
                        <div class="session-metric-card">
                            <span>Average Response</span>
                            <strong>{avg_response_text}</strong>
                        </div>
                        <div class="session-metric-card">
                            <span>Success Rate</span>
                            <strong>{success_text}</strong>
                        </div>
                        <div class="session-metric-card">
                            <span>First Request</span>
                            <strong>{format_timestamp(session_row["first_request"])}</strong>
                        </div>
                        <div class="session-metric-card">
                            <span>Last Request</span>
                            <strong>{format_timestamp(session_row["last_request"])}</strong>
                        </div>
                    </div>

                    <div class="session-section">
                        <div class="grid session-counts-grid">
                            <div>
                                <h3>Persisted Cumulative Counts</h3>
                                <div class="cumulative-counts">{cumulative_counts_html}</div>
                            </div>
                            <div>
                                <h3>API Call Distribution</h3>
                                <div class="cumulative-counts">{api_call_distribution_html}</div>
                            </div>
                        </div>
                    </div>

                    <div class="session-section">
                        <h3>Requests</h3>
                        <p class="section-description">
                            Requests are shown newest first. Duplicate images are retained.
                            Showing 20 requests per page.
                        </p>
                        {request_pages_html}
                    </div>
                </div>
            </div>
            """
        )

    return session_cards, session_detail_sections


CSS = """
* { box-sizing: border-box; }

:root {
    --page-bg: #000;
    --surface: #05070d;
    --surface-raised: #090d16;
    --surface-soft: #0f172a;
    --text: #f8fafc;
    --muted: #9ca3af;
    --muted-strong: #cbd5e1;
    --border: #1f2937;
    --blue: #2563eb;
    --blue-bright: #38bdf8;
    --blue-glow: rgba(37, 99, 235, 0.38);
    --danger-bg: #2a0f14;
    --danger-text: #fca5a5;
    --success-bg: #052e1a;
    --success-text: #86efac;
}

body {
    margin: 0;
    padding: 30px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 34rem), var(--page-bg);
    color: var(--text);
}

.dashboard { max-width: 1600px; margin: 0 auto; }

h1 { margin-top: 0; margin-bottom: 8px; font-size: 32px; }
h2, h3 { margin-top: 0; }

.subtitle { margin-bottom: 30px; color: var(--muted); }

.section { margin-bottom: 20px; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    min-width: 0;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.section-title { margin-top: 0; margin-bottom: 6px; font-size: 22px; }
.section-description { color: var(--muted); margin-top: 0; margin-bottom: 20px; }

.latest-sessions-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
}

.session-card {
    width: 100%;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    font-family: inherit;
}

.session-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 0 3px var(--blue-glow), 0 18px 35px rgba(0, 0, 0, 0.55);
    border-color: var(--blue);
}

.session-card.active {
    border-color: var(--blue-bright);
    box-shadow: 0 0 0 3px var(--blue-glow), 0 18px 35px rgba(0, 0, 0, 0.55);
}

.session-card-title { font-size: 18px; font-weight: 700; word-break: break-all; margin-bottom: 18px; }

.session-card-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }

.session-card-metrics span, .session-metric-card span {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 5px;
}

.session-card-metrics strong { font-size: 16px; }

.session-card-counts { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); }
.session-card-counts > span { display: block; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.session-card-counts .cumulative-counts { max-width: none; }

.session-card-footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 13px; color: var(--muted); }

.session-detail {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}

.session-detail.active {
    display: block;
    border-color: var(--blue);
    box-shadow: 0 0 0 3px var(--blue-glow), 0 18px 50px rgba(0, 0, 0, 0.5);
}

.session-detail-label { font-size: 13px; color: var(--muted); margin-bottom: 5px; }

.session-detail-toggle {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    margin-bottom: 25px;
    padding: 16px 18px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-raised);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    text-align: left;
    transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}

.session-detail-toggle:hover { background: var(--surface-soft); border-color: var(--blue); box-shadow: 0 0 0 3px var(--blue-glow); }
.session-detail-toggle h2 { margin: 0; }

.session-metric-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 30px;
}

.session-metric-card { border: 1px solid var(--border); border-radius: 10px; padding: 15px; background: var(--surface-raised); }
.session-metric-card strong { display: block; font-size: 15px; word-break: break-word; }

.session-section { margin-top: 30px; }

.cumulative-counts { max-width: 500px; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

.count-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--border); }
.count-row:last-child { border-bottom: none; }

.muted { color: var(--muted); }

.request-pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin-top: 25px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}

.pagination-button {
    border: 1px solid var(--border);
    background: var(--surface-raised);
    color: var(--text);
    border-radius: 8px;
    padding: 9px 16px;
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    min-width: 110px;
}

.pagination-button:hover:not(:disabled) { background: var(--surface-soft); border-color: var(--blue); }
.pagination-button:disabled { opacity: 0.45; cursor: not-allowed; }

.pagination-status { min-width: 70px; text-align: center; font-size: 14px; font-weight: 600; color: var(--muted-strong); }

.request-card-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }

.request-card { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--surface); }
.request-card:hover { border-color: var(--blue); box-shadow: 0 0 0 3px var(--blue-glow); }

.request-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 15px;
    padding: 16px;
    border-bottom: 1px solid var(--border);
}

.request-image-name { font-weight: 700; word-break: break-all; }
.request-timestamp { margin-top: 5px; color: var(--muted); font-size: 12px; }

.status-badge { flex-shrink: 0; padding: 5px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.status-success { background: var(--success-bg); color: var(--success-text); }
.status-failed { background: var(--danger-bg); color: var(--danger-text); }

.result-image-container {
    width: min(5in, 100%);
    height: min(5in, 100%);
    aspect-ratio: 1 / 1;
    margin: 0 auto;
    background: var(--surface-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.result-image {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.result-image-unavailable {
    width: min(5in, 100%);
    height: min(5in, 100%);
    aspect-ratio: 1 / 1;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px;
    background: var(--surface-soft);
    color: var(--muted);
    text-align: center;
}

.request-details { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; padding: 15px; border-bottom: 1px solid var(--border); }
.detail-item span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }
.detail-item strong { display: block; font-size: 13px; word-break: break-word; }

.request-counts { padding: 15px; }
.counts-title { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
.request-counts .count-row { padding: 7px 0; font-size: 13px; }

.error-message { margin: 0 15px 15px; padding: 10px; border-radius: 8px; background: var(--danger-bg); color: var(--danger-text); font-size: 13px; word-break: break-word; }

.empty-state { padding: 30px; text-align: center; color: var(--muted); background: var(--surface-raised); border: 1px solid var(--border); border-radius: 10px; }

@media (max-width: 1300px) {
    .session-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}

@media (max-width: 1000px) {
    .latest-sessions-grid { grid-template-columns: 1fr; }
    .request-card-grid { grid-template-columns: 1fr; }
    .grid { grid-template-columns: 1fr; }
}

@media (max-width: 700px) {
    body { padding: 15px; }
    .session-metric-grid { grid-template-columns: 1fr 1fr; }
    .request-details { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 500px) {
    .session-metric-grid, .request-details { grid-template-columns: 1fr; }
    .request-pagination { gap: 8px; }
    .pagination-button { min-width: 90px; padding: 8px 10px; }
}
"""

SCRIPT = """
function showSession(sessionId) {
    const details = document.querySelectorAll(".session-detail");
    const cards = document.querySelectorAll(".session-card");

    details.forEach(function(detail) { detail.classList.remove("active"); });

    cards.forEach(function(card) {
        card.classList.toggle("active", card.dataset.sessionTarget === sessionId);
    });

    const selected = document.getElementById(sessionId);

    if (selected) {
        selected.classList.add("active");
        selected.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function closeSession(sessionId) {
    const selected = document.getElementById(sessionId);
    const card = document.querySelector('.session-card[data-session-target="' + sessionId + '"]');

    if (selected) { selected.classList.remove("active"); }
    if (card) { card.classList.remove("active"); }
}

function changeRequestPage(paginationId, direction) {
    const pagination = document.getElementById(paginationId);
    if (!pagination) { return; }

    const currentPageElement = pagination.querySelector(".pagination-status");
    const currentPage = parseInt(currentPageElement.dataset.currentPage);
    const totalPages = parseInt(currentPageElement.dataset.totalPages);

    let newPage = currentPage + direction;
    if (newPage < 1) { newPage = 1; }
    if (newPage > totalPages) { newPage = totalPages; }
    if (newPage === currentPage) { return; }

    const sessionContainer = pagination.previousElementSibling;
    const pages = sessionContainer.querySelectorAll(".request-page");

    pages.forEach(function(page, index) {
        page.style.display = index === newPage - 1 ? "block" : "none";
    });

    currentPageElement.dataset.currentPage = newPage;
    currentPageElement.textContent = newPage + " / " + totalPages;

    const buttons = pagination.querySelectorAll(".pagination-button");
    buttons[0].disabled = newPage === 1;
    buttons[1].disabled = newPage === totalPages;

    sessionContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.addEventListener("DOMContentLoaded", function() {
    const paginations = document.querySelectorAll(".request-pagination");

    paginations.forEach(function(pagination) {
        const status = pagination.querySelector(".pagination-status");
        const buttons = pagination.querySelectorAll(".pagination-button");
        const totalPages = parseInt(status.dataset.totalPages);

        buttons[0].disabled = true;
        buttons[1].disabled = totalPages <= 1;
    });
});
"""


def generate_dashboard():
    requests_df = load_requests_df()

    if requests_df.empty:
        html_output = empty_dashboard_html(
            "Session Monitoring Dashboard",
            "No request history is available yet.",
        )
        return write_dashboard(OUTPUT_FILENAME, html_output)

    session_counters = load_session_counters()
    session_summary = build_session_summary(requests_df)

    session_cards, session_detail_sections = build_session_sections(
        requests_df, session_summary, session_counters
    )

    html_output = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session Monitoring Dashboard</title>
<style>
{CSS}
</style>
<script>
{SCRIPT}
</script>
</head>
<body>
<div class="dashboard">

<h1>Session Monitoring Dashboard</h1>
<p class="subtitle">Per-session debugging: request history, annotated results, and persisted counts.</p>

<div class="card section">
    <h2 class="section-title">Latest {LATEST_SESSION_COUNT} Sessions</h2>
    <p class="section-description">
        Select a session to view its metrics, persisted cumulative counts, and request results.
    </p>
    <div class="latest-sessions-grid">
        {"".join(session_cards)}
    </div>
</div>

{"".join(session_detail_sections)}

</div>
</body>
</html>
"""

    return write_dashboard(OUTPUT_FILENAME, html_output)


if __name__ == "__main__":
    generate_dashboard()
