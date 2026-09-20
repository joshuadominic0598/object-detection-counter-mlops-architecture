"""Generate the service + model observability dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from helpers import apply_chart_theme, empty_dashboard_html, format_timestamp, load_requests_df, safe_text, write_dashboard


OUTPUT_FILENAME = "monitoring_dashboard.html"
LATEST_SESSION_COUNT = 5
MAX_SESSION_COUNT = 20


def _is_missing(value):
    try:
        return pd.isna(value)
    except (TypeError, ValueError):
        return value is None


def _format_ms(value):
    return "N/A" if _is_missing(value) else f"{float(value):.0f} ms"


def _format_percent(value):
    return "N/A" if _is_missing(value) else f"{float(value):.1f}%"


def _safe_quantile(series, quantile):
    clean_series = series.dropna()

    if clean_series.empty:
        return float("nan")

    return float(clean_series.quantile(quantile))


def _empty_chart(title, message):
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font=dict(color="#9ca3af", size=14),
    )
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=20, r=20, t=60, b=20))
    return apply_chart_theme(fig)


def _bar_text(values):
    return [f"{int(value):,}" if pd.notna(value) else "" for value in values]


def _style_bar_chart(fig, text_values):
    fig.update_traces(
        text=text_values,
        textposition="outside",
        marker_line_width=4,
        marker_line_color="rgba(255, 255, 255, 0.28)",
    )
    fig.update_layout(bargap=0.2, bargroupgap=0.08)
    return fig


def _build_class_counts(requests_df):
    rows = []

    for _, row in requests_df.iterrows():
        counts = row.get("counts")
        timestamp = row.get("request_timestamp")

        if not isinstance(counts, dict):
            continue

        for class_name, count_value in counts.items():
            try:
                count_value = int(count_value)
            except (ValueError, TypeError):
                continue

            if count_value <= 0:
                continue

            rows.append(
                {
                    "request_timestamp": timestamp,
                    "class_name": str(class_name),
                    "count": count_value,
                }
            )

    return pd.DataFrame(rows)


def _build_zero_result_requests(requests_df):
    zero_result_df = requests_df[requests_df["detected_count"].fillna(0) == 0].copy()

    if zero_result_df.empty:
        return zero_result_df

    zero_result_df = zero_result_df.sort_values("request_timestamp", ascending=False)
    zero_result_df["request_timestamp"] = zero_result_df["request_timestamp"].map(format_timestamp)
    zero_result_df["threshold"] = zero_result_df["threshold"].map(
        lambda value: "N/A" if _is_missing(value) else f"{float(value):.2f}"
    )
    zero_result_df["response_time_ms"] = zero_result_df["response_time_ms"].map(_format_ms)

    return zero_result_df


def build_session_summary(requests_df):
    session_summary = (
        requests_df.groupby("session_id")
        .agg(
            requests=("request_id", "count"),
            detections=("detected_count", "sum"),
            detection_rate=("detected_count", lambda series: (series.fillna(0) > 0).mean() * 100),
            zero_result_rate=("detected_count", lambda series: (series.fillna(0) == 0).mean() * 100),
            avg_objects_per_request=("detected_count", "mean"),
            avg_response_ms=("response_time_ms", "mean"),
            p95_latency_ms=("response_time_ms", lambda series: _safe_quantile(series, 0.95)),
            success_rate=("success", "mean"),
            first_request=("request_timestamp", "min"),
            last_request=("request_timestamp", "max"),
        )
        .reset_index()
    )

    session_summary["success_rate"] = session_summary["success_rate"] * 100
    session_summary["detection_rate"] = session_summary["detection_rate"].round(2)
    session_summary["zero_result_rate"] = session_summary["zero_result_rate"].round(2)
    session_summary["avg_objects_per_request"] = session_summary["avg_objects_per_request"].round(2)
    session_summary["avg_response_ms"] = session_summary["avg_response_ms"].round(2)
    session_summary["p95_latency_ms"] = session_summary["p95_latency_ms"].round(2)
    session_summary["success_rate"] = session_summary["success_rate"].round(2)
    session_summary["detections"] = session_summary["detections"].astype(int)

    return session_summary.sort_values("last_request", ascending=False)


def _build_time_series(requests_df):
    timestamped_df = requests_df[requests_df["request_timestamp"].notna()].copy()

    if timestamped_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    timeline = timestamped_df.set_index("request_timestamp")
    day_index = pd.date_range(
        start=timeline.index.min().normalize(),
        end=timeline.index.max().normalize(),
        freq="D",
    )
    if len(day_index) > 8:
        day_index = day_index[-8:]

    daily_timeline = timeline.loc[day_index.min() : day_index.max()]

    request_volume_df = (
        daily_timeline.resample("D").size().reindex(day_index, fill_value=0).rename_axis("request_timestamp").reset_index(name="requests")
    )
    response_time_df = (
        daily_timeline["response_time_ms"]
        .resample("D")
        .mean()
        .reindex(day_index)
        .rename_axis("request_timestamp")
        .reset_index(name="response_time_ms")
    )
    error_rate_df = (
        daily_timeline["success"]
        .resample("D")
        .apply(lambda series: (1 - series.mean()) * 100)
        .reindex(day_index)
        .rename_axis("request_timestamp")
        .reset_index(name="error_rate")
    )

    p50_df = (
        daily_timeline["response_time_ms"]
        .resample("D")
        .quantile(0.50)
        .reindex(day_index)
        .rename_axis("request_timestamp")
        .reset_index(name="p50")
    )
    p95_df = (
        daily_timeline["response_time_ms"]
        .resample("D")
        .quantile(0.95)
        .reindex(day_index)
        .rename_axis("request_timestamp")
        .reset_index(name="p95")
    )
    p99_df = (
        daily_timeline["response_time_ms"]
        .resample("D")
        .quantile(0.99)
        .reindex(day_index)
        .rename_axis("request_timestamp")
        .reset_index(name="p99")
    )

    latency_df = p50_df.merge(p95_df, on="request_timestamp", how="outer").merge(p99_df, on="request_timestamp", how="outer")

    return timeline, request_volume_df, response_time_df, error_rate_df, latency_df


def build_charts(requests_df):
    timeline, request_volume_df, response_time_df, error_rate_df, latency_df = _build_time_series(requests_df)
    class_counts_df = _build_class_counts(requests_df)

    if request_volume_df.empty:
        return {
            "request_volume": _empty_chart("Request Volume", "No request history available"),
            "response_time": _empty_chart("Response Time", "No request history available"),
            "error_rate": _empty_chart("Error Rate", "No request history available"),
            "latency_percentiles": _empty_chart("Latency Percentiles", "No request history available"),
            "detections_by_class": _empty_chart("Detections by Class", "No detections available"),
            "class_distribution": _empty_chart("Class Distribution Over Time", "No detections available"),
            "detection_outcome": _empty_chart("Detection Outcome", "No request history available"),
            "threshold_sensitivity": _empty_chart("Threshold Sensitivity", "No threshold data available"),
        }

    request_volume_fig = apply_chart_theme(
        px.bar(request_volume_df, x="request_timestamp", y="requests", title="Request Volume")
    )
    request_volume_fig.update_layout(xaxis_title="Day", yaxis_title="Requests", hovermode="x unified")
    _style_bar_chart(request_volume_fig, _bar_text(request_volume_df["requests"]))

    response_time_fig = apply_chart_theme(
        px.bar(response_time_df, x="request_timestamp", y="response_time_ms", title="Response Time")
    )
    response_time_fig.update_layout(xaxis_title="Day", yaxis_title="Milliseconds", hovermode="x unified")
    _style_bar_chart(response_time_fig, _bar_text(response_time_df["response_time_ms"]))

    error_rate_fig = apply_chart_theme(
        px.line(
            error_rate_df,
            x="request_timestamp",
            y="error_rate",
            markers=True,
            title="Error Rate",
        )
    )
    error_rate_fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Error Rate (%)",
        hovermode="x unified",
        yaxis=dict(rangemode="tozero"),
    )
    error_rate_fig.update_traces(line=dict(width=4, color="#f97316"), marker=dict(size=9, color="#f97316"))

    latency_long_df = latency_df.melt(
        id_vars="request_timestamp",
        value_vars=["p50", "p95", "p99"],
        var_name="percentile",
        value_name="latency_ms",
    )
    latency_percentiles_fig = apply_chart_theme(
        px.bar(
            latency_long_df,
            x="request_timestamp",
            y="latency_ms",
            color="percentile",
            barmode="group",
            title="Latency Percentiles",
        )
    )
    latency_percentiles_fig.update_layout(xaxis_title="Day", yaxis_title="Milliseconds", hovermode="x unified")
    _style_bar_chart(
        latency_percentiles_fig,
        latency_long_df["latency_ms"].map(lambda value: "" if pd.isna(value) else f"{value:.0f}"),
    )

    if class_counts_df.empty:
        detections_by_class_fig = _empty_chart("Detections by Class", "No detections available")
        class_distribution_fig = _empty_chart("Class Distribution Over Time", "No detections available")
    else:
        class_totals_df = (
            class_counts_df.groupby("class_name", as_index=False)["count"].sum().sort_values("count", ascending=True)
        )
        detections_by_class_fig = apply_chart_theme(
            px.bar(class_totals_df, x="count", y="class_name", orientation="h", title="Detections by Class")
        )
        detections_by_class_fig.update_layout(xaxis_title="Detections", yaxis_title="Class")

        class_distribution_df = (
            class_counts_df.set_index("request_timestamp")
            .groupby("class_name")["count"]
            .resample("1h")
            .sum()
            .reset_index()
        )
        top_classes = class_totals_df.sort_values("count", ascending=False).head(2)["class_name"].tolist()
        class_distribution_df = class_distribution_df[class_distribution_df["class_name"].isin(top_classes)]

        class_distribution_fig = apply_chart_theme(
            px.line(
                class_distribution_df,
                x="request_timestamp",
                y="count",
                color="class_name",
                markers=True,
                title="Class Distribution Over Time",
            )
        )
        class_distribution_fig.update_layout(xaxis_title="Time", yaxis_title="Detections", hovermode="x unified")
        class_distribution_fig.update_traces(mode="lines+markers", line=dict(width=4), marker=dict(size=10))

    outcome_df = requests_df.assign(
        outcome=requests_df["detected_count"].fillna(0).map(lambda value: "Detected" if value > 0 else "Zero result")
    )
    outcome_df = outcome_df["outcome"].value_counts().reset_index()
    outcome_df.columns = ["outcome", "requests"]
    detection_outcome_fig = apply_chart_theme(
        px.pie(outcome_df, names="outcome", values="requests", hole=0.45, title="Detection Outcome")
    )
    detection_outcome_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))

    threshold_df = requests_df[requests_df["threshold"].notna()].copy()

    if threshold_df.empty:
        threshold_sensitivity_fig = _empty_chart("Threshold Sensitivity", "No threshold data available")
    else:
        threshold_df["threshold_label"] = threshold_df["threshold"].map(lambda value: f"{float(value):.2f}")
        threshold_summary_df = (
            threshold_df.groupby("threshold_label", as_index=False)
            .agg(
                requests=("request_id", "count"),
                detection_rate=("detected_count", lambda series: (series.fillna(0) > 0).mean() * 100),
                avg_detections=("detected_count", "mean"),
            )
            .sort_values("threshold_label")
        )
        threshold_sensitivity_fig = apply_chart_theme(
            px.bar(
                threshold_summary_df,
                x="threshold_label",
                y="requests",
                title="Confidence Distribution / Threshold Sensitivity",
                hover_data={"detection_rate": ":.1f", "avg_detections": ":.2f"},
            )
        )
        threshold_sensitivity_fig.update_layout(xaxis_title="Confidence Threshold", yaxis_title="Requests")
        threshold_sensitivity_fig.update_traces(
            text=threshold_summary_df["requests"].map(lambda value: f"{int(value):,}"),
            textposition="outside",
        )

    return {
        "request_volume": request_volume_fig,
        "response_time": response_time_fig,
        "error_rate": error_rate_fig,
        "latency_percentiles": latency_percentiles_fig,
        "detections_by_class": detections_by_class_fig,
        "class_distribution": class_distribution_fig,
        "detection_outcome": detection_outcome_fig,
        "threshold_sensitivity": threshold_sensitivity_fig,
    }


def _metric_card(label, value, hint=""):
    hint_html = f'<div class="metric-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {hint_html}
    </div>
    """


def _build_kpi_rows(metrics):
    top_row = "".join(
        [
            _metric_card("REQUESTS", f"{metrics['total_requests']:,}"),
            _metric_card("SUCCESS RATE", _format_percent(metrics["success_rate"])),
            _metric_card("DETECTIONS", f"{metrics['total_detections']:,}"),
            _metric_card("P95 LATENCY", _format_ms(metrics["p95_latency_ms"])),
        ]
    )

    bottom_row = "".join(
        [
            _metric_card("SESSIONS", f"{metrics['total_sessions']:,}"),
            _metric_card("DETECTION RATE", _format_percent(metrics["detection_rate"])),
            _metric_card("ZERO RESULT RATE", _format_percent(metrics["zero_result_rate"])),
            _metric_card("AVG OBJ/REQ", f"{metrics['avg_objects_per_request']:.2f}"),
        ]
    )

    return top_row, bottom_row


def _build_zero_result_requests_card(zero_result_df):
    if zero_result_df.empty:
        return """
        <div class="stat-card">
            <div class="stat-title">Zero Result Request Details</div>
            <div class="empty-state">No zero-result requests available.</div>
        </div>
        """

    display_df = zero_result_df[["session_id", "request_timestamp", "threshold", "response_time_ms"]].copy()
    display_df.columns = ["Session ID", "Request Time", "Threshold", "Response Time"]

    return f"""
    <div class="stat-card">
        <div class="stat-title">Zero Result Request Details</div>
        <div class="zero-result-table-wrap">
            {display_df.to_html(index=False, classes="zero-result-table", escape=False)}
        </div>
    </div>
    """


def _build_sessions_section(session_summary):
    latest_sessions = session_summary.head(MAX_SESSION_COUNT)
    session_rows = []

    for index, (_, session) in enumerate(latest_sessions.iterrows()):
        visible_class = "" if index < LATEST_SESSION_COUNT else "session-row-hidden"
        session_rows.append(
            f"""
            <tr class="session-row {visible_class}">
                <td>{safe_text(session['session_id'])}</td>
                <td>{int(session['requests']):,}</td>
                <td>{int(session['detections']):,}</td>
                <td>{_format_percent(session['detection_rate'])}</td>
                <td>{_format_ms(session['p95_latency_ms'])}</td>
                <td>{format_timestamp(session['last_request'])}</td>
            </tr>
            """
        )

    if not session_rows:
        table_body = '<tr><td colspan="6">No session history is available yet.</td></tr>'
    else:
        table_body = "".join(session_rows)

    return f"""
    <div class="session-controls">
        <label for="session-limit">Show</label>
        <select id="session-limit" class="session-limit-select" onchange="setSessionLimit(this.value)">
            <option value="5" selected>5</option>
            <option value="10">10</option>
            <option value="15">15</option>
            <option value="20">20</option>
        </select>
        <span>sessions</span>
    </div>
    <div class="summary-table-wrap">
        <table class="session-table">
            <thead>
                <tr>
                    <th>Session ID</th>
                    <th>Requests</th>
                    <th>Detections</th>
                    <th>Rate</th>
                    <th>P95</th>
                    <th>Last Seen</th>
                </tr>
            </thead>
            <tbody id="session-table-body">
                {table_body}
            </tbody>
        </table>
    </div>
    """


def _build_session_summary_table(session_summary):
    table_df = session_summary.copy()
    table_df["Session ID"] = table_df["session_id"].map(str)
    table_df["Requests"] = table_df["requests"].map(lambda value: f"{int(value):,}")
    table_df["Detections"] = table_df["detections"].map(lambda value: f"{int(value):,}")
    table_df["Rate"] = table_df["detection_rate"].map(_format_percent)
    table_df["P95"] = table_df["p95_latency_ms"].map(_format_ms)
    table_df["Last Seen"] = table_df["last_request"].map(format_timestamp)

    table_df = table_df[["Session ID", "Requests", "Detections", "Rate", "P95", "Last Seen"]]

    return table_df.to_html(index=False, classes="session-table", escape=False)


def _build_latest_sessions_cards(session_summary):
    latest_sessions = session_summary.head(LATEST_SESSION_COUNT)

    if latest_sessions.empty:
        return '<div class="empty-state">No session history is available yet.</div>'

    cards = []

    for _, session in latest_sessions.iterrows():
        cards.append(
            f"""
            <article class="investigation-card">
                <div class="investigation-card-header">
                    <div>
                        <div class="investigation-session-id">{safe_text(session['session_id'])}</div>
                        <div class="investigation-session-meta">Last seen {format_timestamp(session['last_request'])}</div>
                    </div>
                    <div class="investigation-pill">{_format_percent(session['detection_rate'])}</div>
                </div>
                <div class="investigation-card-grid">
                    <div><span>Requests</span><strong>{int(session['requests']):,}</strong></div>
                    <div><span>Detections</span><strong>{int(session['detections']):,}</strong></div>
                    <div><span>P95</span><strong>{_format_ms(session['p95_latency_ms'])}</strong></div>
                    <div><span>Zero</span><strong>{_format_percent(session['zero_result_rate'])}</strong></div>
                </div>
            </article>
            """
        )

    return "".join(cards)


CSS = """
* { box-sizing: border-box; }

:root {
    --page-bg: #02040a;
    --surface: rgba(8, 13, 24, 0.9);
    --surface-strong: #0b1220;
    --surface-soft: rgba(15, 23, 42, 0.72);
    --text: #f8fafc;
    --muted: #94a3b8;
    --muted-strong: #cbd5e1;
    --border: rgba(148, 163, 184, 0.18);
    --accent: #38bdf8;
    --accent-strong: #2563eb;
    --accent-glow: rgba(56, 189, 248, 0.18);
    --accent-glow-strong: rgba(37, 99, 235, 0.25);
    --success: #22c55e;
    --warning: #f59e0b;
}

body {
    margin: 0;
    padding: 28px;
    color: var(--text);
    font-family: "SF Pro Display", "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background:
        radial-gradient(circle at 12% 12%, rgba(56, 189, 248, 0.18), transparent 22rem),
        radial-gradient(circle at 88% 14%, rgba(37, 99, 235, 0.16), transparent 24rem),
        linear-gradient(180deg, #030712 0%, #02040a 100%);
}

.dashboard {
    max-width: 1600px;
    margin: 0 auto;
}

.hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 20px;
    margin-bottom: 24px;
}

.hero-panel,
.mini-panel,
.panel,
.metric-card,
.stat-card,
.investigation-card {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(8, 13, 24, 0.96));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(16px);
}

.hero-panel {
    border-radius: 26px;
    padding: 30px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    background: rgba(56, 189, 248, 0.08);
    color: var(--accent);
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 18px 0 10px;
    font-size: clamp(34px, 4.2vw, 54px);
    line-height: 1;
    letter-spacing: -0.04em;
}

.subtitle {
    margin: 0;
    max-width: 70ch;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.6;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 22px;
}

.hero-badge {
    padding: 10px 14px;
    border-radius: 14px;
    border: 1px solid rgba(56, 189, 248, 0.18);
    background: rgba(15, 23, 42, 0.72);
    color: var(--muted-strong);
    font-size: 13px;
}

.metric-strip,
.session-grid,
.chart-grid,
.latest-grid {
    display: grid;
    gap: 16px;
}

.metric-strip {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-bottom: 16px;
}

.metric-card {
    border-radius: 20px;
    padding: 20px;
    min-width: 0;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), transparent 45%);
    pointer-events: none;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.metric-hint {
    margin-top: 8px;
    color: var(--muted);
    font-size: 13px;
}

.panel {
    border-radius: 24px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-head {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 16px;
    margin-bottom: 18px;
}

.section-title {
    margin: 0;
    font-size: 22px;
    letter-spacing: -0.03em;
}

.section-description {
    margin: 6px 0 0;
    color: var(--muted);
}

.chart-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.chart-card,
.stat-card {
    border-radius: 20px;
    padding: 18px;
    min-width: 0;
}

.chart-card {
    border: 1px solid var(--border);
    background: var(--surface);
}

.chart-card.full-width {
    grid-column: 1 / -1;
}

.chart-card .js-plotly-plot,
.chart-card .plotly-graph-div {
    width: 100% !important;
}

.chart-card .plot-container,
.chart-card .svg-container {
    width: 100% !important;
}

.zero-result-table-wrap {
    overflow-x: auto;
    max-height: 420px;
    overflow-y: auto;
}

.zero-result-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(15, 23, 42, 0.9);
}

.zero-result-table th,
.zero-result-table td {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    text-align: left;
    white-space: nowrap;
}

.zero-result-table th {
    color: var(--muted-strong);
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-full {
    width: 100%;
}

.stat-card {
    border: 1px solid rgba(56, 189, 248, 0.18);
    background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(8, 13, 24, 0.9));
}

.stat-title {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.stat-main {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.05em;
    margin-bottom: 14px;
}

.stat-subgrid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.stat-subgrid span,
.investigation-card-grid span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.stat-subgrid strong,
.investigation-card-grid strong {
    font-size: 18px;
}

.summary-table-wrap {
    overflow-x: auto;
    border-radius: 18px;
    border: 1px solid var(--border);
}

.session-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface-strong);
}

.session-table th,
.session-table td {
    padding: 14px 16px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    text-align: left;
    white-space: nowrap;
}

.session-table th {
    background: rgba(15, 23, 42, 0.92);
    color: var(--muted-strong);
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.session-table tbody tr:hover {
    background: rgba(56, 189, 248, 0.05);
}

.latest-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.investigation-card {
    border-radius: 20px;
    padding: 18px;
}

.investigation-card-header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: start;
    margin-bottom: 16px;
}

.investigation-session-id {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.investigation-session-meta {
    color: var(--muted);
    margin-top: 4px;
    font-size: 13px;
}

.investigation-pill {
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.18);
    color: #86efac;
    font-weight: 600;
}

.investigation-card-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.investigation-card-footer {
    display: flex;
    justify-content: flex-end;
    margin-top: 16px;
}

.investigation-link {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
}

.session-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    color: var(--muted);
}

.session-limit-select {
    background: rgba(15, 23, 42, 0.9);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 10px;
}

.session-row-hidden {
    display: none;
}

.section-header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-action-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.2);
    background: rgba(56, 189, 248, 0.08);
    color: var(--accent);
    text-decoration: none;
    font-weight: 700;
    white-space: nowrap;
}

.empty-state {
    border: 1px dashed rgba(148, 163, 184, 0.24);
    border-radius: 18px;
    padding: 28px;
    color: var(--muted);
    text-align: center;
    background: rgba(15, 23, 42, 0.42);
}

@media (max-width: 1200px) {
    .hero,
    .metric-strip,
    .latest-grid,
    .chart-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 800px) {
    body { padding: 16px; }
    .hero-panel,
    .mini-panel,
    .panel,
    .metric-card,
    .stat-card,
    .investigation-card {
        border-radius: 18px;
    }

    .panel-head,
    .investigation-card-header {
        flex-direction: column;
        align-items: start;
    }

    .metric-value,
    .stat-main {
        font-size: 30px;
    }
}
"""


def generate_dashboard():
    requests_df = load_requests_df()

    if requests_df.empty:
        html_output = empty_dashboard_html(
            "Object Detection Monitoring",
            "No request history is available yet.",
        )
        return write_dashboard(OUTPUT_FILENAME, html_output)

    metrics = {
        "total_requests": len(requests_df),
        "total_sessions": requests_df["session_id"].nunique(),
        "total_detections": int(requests_df["detected_count"].fillna(0).sum()),
        "success_rate": requests_df["success"].mean() * 100,
        "p95_latency_ms": _safe_quantile(requests_df["response_time_ms"], 0.95),
        "detection_rate": (requests_df["detected_count"].fillna(0) > 0).mean() * 100,
        "zero_result_rate": (requests_df["detected_count"].fillna(0) == 0).mean() * 100,
        "avg_objects_per_request": requests_df["detected_count"].fillna(0).mean(),
        "p50_objects_per_request": requests_df["detected_count"].fillna(0).quantile(0.50),
        "p95_objects_per_request": requests_df["detected_count"].fillna(0).quantile(0.95),
        "max_objects_per_request": requests_df["detected_count"].fillna(0).max(),
    }

    zero_result_df = _build_zero_result_requests(requests_df)
    charts = build_charts(requests_df)
    session_summary = build_session_summary(requests_df)
    latest_sessions_html = _build_latest_sessions_cards(session_summary)
    sessions_section_html = _build_sessions_section(session_summary)
    top_row_metrics_html, bottom_row_metrics_html = _build_kpi_rows(metrics)
    zero_result_requests_html = _build_zero_result_requests_card(zero_result_df)

    request_volume_chart_html = charts["request_volume"].to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})
    response_time_chart_html = charts["response_time"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    error_rate_chart_html = charts["error_rate"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    latency_percentiles_chart_html = charts["latency_percentiles"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    detections_by_class_chart_html = charts["detections_by_class"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    class_distribution_chart_html = charts["class_distribution"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    detection_outcome_chart_html = charts["detection_outcome"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    threshold_sensitivity_chart_html = charts["threshold_sensitivity"].to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    html_output = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Detection Monitoring</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="dashboard">
    <section class="hero">
        <div class="hero-panel hero-full">
            <div class="eyebrow">Service + Model Observability</div>
            <h1>Object Detection Monitoring</h1>
            <p class="subtitle">
                Live service health, model behavior, and session investigation in one dashboard.
                The dashboard only reflects annotated debug images and request telemetry produced by the single API route.
            </p>
            <div class="hero-meta">
                <div class="hero-badge">Current route: /object-detection</div>
                <div class="hero-badge">Annotated images: results/&lt;session_id&gt;/</div>
                <div class="hero-badge">Investigate details in session_dashboard.html</div>
            </div>
        </div>
    </section>

    <section class="metric-strip">
        {top_row_metrics_html}
    </section>

    <section class="metric-strip">
        {bottom_row_metrics_html}
    </section>

    <section class="panel">
        <div class="panel-head">
            <div>
                <h2 class="section-title">Service Health</h2>
                <p class="section-description">Request volume, response time, errors, and latency percentiles.</p>
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">{request_volume_chart_html}</div>
            <div class="chart-card">{response_time_chart_html}</div>
            <div class="chart-card">{error_rate_chart_html}</div>
            <div class="chart-card">{latency_percentiles_chart_html}</div>
        </div>
    </section>

    <section class="panel">
        <div class="panel-head">
            <div>
                <h2 class="section-title">Model Behavior</h2>
                <p class="section-description">Detected classes, class distribution over time, outcome balance, and threshold usage.</p>
            </div>
        </div>
        <div class="chart-grid">
            <div class="chart-card">{detections_by_class_chart_html}</div>
            <div class="chart-card">{class_distribution_chart_html}</div>
            <div class="chart-card">{detection_outcome_chart_html}</div>
            <div class="chart-card">{zero_result_requests_html}</div>
            <div class="chart-card full-width">{threshold_sensitivity_chart_html}</div>
        </div>
    </section>

    <section class="panel">
        <div class="panel-head">
            <div>
                <h2 class="section-title">Investigation</h2>
                <p class="section-description">Latest sessions with a quick path into the annotated request history.</p>
            </div>
            <a class="section-action-button" href="session_dashboard.html">Open latest session</a>
        </div>
        <div class="latest-grid">
            {latest_sessions_html}
        </div>
    </section>

    <section class="panel">
        <div class="panel-head">
            <div>
                <h2 class="section-title">Sessions</h2>
                <p class="section-description">Operational session summary sorted by recency.</p>
            </div>
        </div>
        {sessions_section_html}
    </section>
</div>
<script>
function setSessionLimit(limit) {{
    const rows = document.querySelectorAll('#session-table-body .session-row');
    const maxVisible = parseInt(limit, 10) || 5;

    rows.forEach((row, index) => {{
        row.classList.toggle('session-row-hidden', index >= maxVisible);
    }});
}}

document.addEventListener('DOMContentLoaded', () => {{
    const selector = document.getElementById('session-limit');

    if (selector) {{
        setSessionLimit(selector.value);
    }}
}});
</script>
</body>
</html>
"""

    return write_dashboard(OUTPUT_FILENAME, html_output)


if __name__ == "__main__":
    generate_dashboard()
