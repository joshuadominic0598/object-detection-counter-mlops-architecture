"""
Generates self-contained HTML dashboards from performance test results
(results/model_management/evaluation/*.json, produced by
model_management/evaluation/evaluator.py against the golden test dataset).

Deliberately minimal and domain-agnostic: this only reports metrics (mAP,
precision, recall, weighted score/F1, per-class counts) for whichever
classes the model was trained on - it does not declare a "winner" or decide
promotion. That decision belongs to a human reading the numbers, or to
model_management/orchestrator.py when run with an explicit --promote-metric.

generate_dashboard() shows the most recent run's best-scoring threshold plus
a compact table of every other threshold tested.

generate_comparison_dashboard(model_names=None) shows the latest recorded
run of each model in `model_names` (defaults to every model currently
downloaded to tmp/models) side by side, at each model's own best threshold.
"""

import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from model_management.evaluation.scoring import IOU_MATCH_THRESHOLD, best_threshold, verdict_tier
from model_management.model_registry.registry import get_active_model_name, is_downloaded, list_models, normalize_model_name

PERFORMANCE_DIR = REPO_ROOT / "results" / "model_management" / "evaluation"
RESULTS_DIR = PERFORMANCE_DIR
OUTPUT_DIR = Path("tmp/dashboard")
OUTPUT_FILE = OUTPUT_DIR / "performance_dashboard.html"
COMPARISON_OUTPUT_FILE = OUTPUT_DIR / "model_comparison_dashboard.html"

CHART_FONT_COLOR = "#cbd5e1"
CHART_TITLE_COLOR = "#f8fafc"
CHART_GRID_COLOR = "#1f2937"

DASHBOARD_STYLES = """
    body {
        font-family: Arial, sans-serif;
        background: #000;
        color: #f8fafc;
        padding: 40px;
        margin: 0;
    }

    .header {
        max-width: 1000px;
        margin: 0 auto 30px auto;
        background: #05070d;
        padding: 24px 32px;
        border: 1px solid #1f2937;
        border-radius: 12px;
    }

    .header h1 {
        margin: 0 0 12px 0;
    }

    .header .model-info {
        color: #9ca3af;
        font-size: 14px;
    }

    .header .model-info strong {
        color: #f8fafc;
    }

    section {
        max-width: 1000px;
        margin: 0 auto 30px auto;
        background: #05070d;
        padding: 24px 32px;
        border: 1px solid #1f2937;
        border-radius: 12px;
    }

    .metric-cards {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 16px 0;
    }

    .metric-card {
        background: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 20px;
        min-width: 140px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 22px;
        font-weight: 700;
        margin-top: 4px;
    }

    table.results-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        font-size: 14px;
    }

    .results-table th, .results-table td {
        text-align: left;
        padding: 8px 12px;
        border-bottom: 1px solid #1f2937;
    }

    .results-table th {
        color: #9ca3af;
        text-transform: uppercase;
        font-size: 11px;
    }

    .cm-note {
        color: #9ca3af;
        font-size: 12px;
        max-width: 700px;
        margin: 0 0 16px 0;
    }

    .tier-ok { color: #22c55e; }
    .tier-warn { color: #eab308; }
    .tier-bad { color: #ef4444; }

    .verdict-callout {
        background: #0f172a;
        border: 1px solid #1f2937;
        border-left: 4px solid #9ca3af;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 14px;
        color: #e5e7eb;
    }

    .verdict-callout.tier-ok { border-left-color: #22c55e; }
    .verdict-callout.tier-warn { border-left-color: #eab308; }
    .verdict-callout.tier-bad { border-left-color: #ef4444; }

    .plots-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }

    .plot-card {
        background: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .plot-card h4 {
        margin: 0 0 10px 0;
        color: #9ca3af;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .plot-card img {
        max-width: 100%;
        border-radius: 6px;
        background: #fff;
    }

    details.guide-section, details.plots-section {
        max-width: 1000px;
        margin: 0 auto 30px auto;
        background: #05070d;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px 32px;
    }

    details.guide-section summary, details.plots-section summary {
        cursor: pointer;
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
    }

    details.guide-section h3 {
        margin: 20px 0 6px 0;
        color: #f8fafc;
        font-size: 15px;
    }

    details.guide-section p {
        color: #9ca3af;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }

    .footnote {
        max-width: 1000px;
        margin: 0 auto 30px auto;
        color: #6b7280;
        font-size: 12px;
        text-align: center;
    }
"""


def apply_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color=CHART_FONT_COLOR, size=14),
        title_font=dict(color=CHART_TITLE_COLOR, size=16),
        legend=dict(font=dict(color=CHART_FONT_COLOR)),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor=CHART_GRID_COLOR,
            font_color=CHART_TITLE_COLOR,
        ),
        margin=dict(t=60, b=40),
    )

    fig.update_xaxes(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR, linecolor=CHART_GRID_COLOR)
    fig.update_yaxes(gridcolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR, linecolor=CHART_GRID_COLOR)

    return fig


def label_bars(fig, fmt="%{text:.1f}"):
    """Show each bar's value on the chart itself so it's readable without hovering."""

    fig.update_traces(texttemplate=fmt, textposition="outside")
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")

    return fig


def safe_text(value):
    if value is None:
        return ""
    return html.escape(str(value))


def image_to_base64(path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


# Kept to just the two most decision-relevant plots - the rest (P curve, R
# curve individually) are redundant with the F1/PR curves and the per-class
# table already shown above them.
PLOT_FILES = [
    ("Normalized Confusion Matrix", "confusion_matrix_normalized.png"),
    ("Precision-Recall Curve", "BoxPR_curve.png"),
]


def build_plots_gallery(plots_dir, title="Confusion matrix & PR curve"):
    """Collapsed by default so the page opens on the numbers, not a wall of
    images; expand to see the underlying plots Ultralytics saved for a run."""

    if not plots_dir:
        return ""

    base_dir = REPO_ROOT / plots_dir
    cards = []

    for plot_title, filename in PLOT_FILES:
        encoded = image_to_base64(base_dir / filename)

        if encoded:
            cards.append(
                f"""
                <div class="plot-card">
                    <h4>{safe_text(plot_title)}</h4>
                    <img src="data:image/png;base64,{encoded}" alt="{safe_text(plot_title)}">
                </div>
                """
            )

    if not cards:
        return ""

    return f"""
    <details class="plots-section">
        <summary>{safe_text(title)}</summary>
        <div class="plots-gallery">{"".join(cards)}</div>
    </details>
    """


def build_verdict_callout(summary):
    """Plain-language read of a single threshold's results - a starting
    point for the reader's own judgment, not a promotion decision."""

    overall = summary.get("overall", {})
    tier, css_class = verdict_tier(overall.get("weighted_score"))

    score = overall.get("weighted_score")
    score_text = f"{score:.1f}%" if score is not None else "N/A"

    per_class = summary.get("per_class") or {}
    scored = {
        name: stats["weighted_score"]
        for name, stats in per_class.items()
        if stats.get("weighted_score") is not None
    }

    class_note = ""

    if len(scored) > 1:
        best_class = max(scored, key=scored.get)
        worst_class = min(scored, key=scored.get)

        if best_class != worst_class:
            class_note = (
                f" '{safe_text(best_class)}' scores highest "
                f"({scored[best_class]:.1f}%); '{safe_text(worst_class)}' scores lowest "
                f"({scored[worst_class]:.1f}%)."
            )

    return f"""
    <div class="verdict-callout {css_class}">
        Overall fit at this threshold is <strong>{tier}</strong> (weighted score {score_text},
        macro-averaged across classes).{class_note}
    </div>
    """


def build_interpretation_guide():
    return f"""
    <details class="guide-section">
        <summary>How to read this dashboard</summary>

        <h3>Precision, recall &amp; weighted score</h3>
        <p>
            Precision is the share of predictions that were correct (fewer false
            alarms means higher precision). Recall is the share of real objects the
            model actually found (fewer misses means higher recall). Weighted score
            here is the F1 score - precision and recall weighted equally, macro-averaged
            across classes. No class is treated as more important than another by this
            dashboard; if one should be, choose that when deciding promotion.
        </p>

        <h3>mAP50 / mAP50-95</h3>
        <p>
            mAP50 measures detection accuracy at a lenient overlap threshold (50%
            IoU). mAP50-95 averages that same measurement across stricter overlap
            thresholds (50%-95%), so it also rewards tightly-fitted bounding boxes,
            not just "roughly the right spot."
        </p>

        <h3>Per-class table</h3>
        <p>
            TP/FP/FN come from Ultralytics' own confusion matrix, where a prediction
            only counts as correct if it overlaps a ground-truth box by at least
            {IOU_MATCH_THRESHOLD*100:.0f}% IoU - "close enough," not just the right
            class anywhere in the image. "Count diff" is total predicted minus total
            ground-truth objects of that class - a useful sanity check, but it can
            mask errors that cancel out (a missed object and a false one balancing to
            a 0 diff), which is why precision/recall/weighted score take priority.
        </p>
    </details>
    """


def default_downloaded_models():
    return [name for name in list_models() if is_downloaded(name)]


def load_runs(model_names=None):
    runs = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if model_names is not None and run.get("model_name") not in model_names:
            continue

        runs.append(run)

    return runs


def _empty_page(title, message):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{safe_text(title)}</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #000; color: #f8fafc; padding: 40px; }}
            .message {{ max-width: 700px; margin: 80px auto; background: #05070d; padding: 40px;
                border: 1px solid #1f2937; border-radius: 12px; text-align: center; }}
            p {{ color: #9ca3af; }}
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


def build_metric_cards(summary):
    overall = summary.get("overall", {})

    metrics = [
        ("mAP50", summary.get("map50"), "{:.1f}%"),
        ("mAP50-95", summary.get("map50_95"), "{:.1f}%"),
        ("Weighted score (F1)", overall.get("weighted_score"), "{:.1f}%"),
        ("Precision", overall.get("precision"), "{:.1f}%"),
        ("Recall", overall.get("recall"), "{:.1f}%"),
        ("Images evaluated", summary.get("images_evaluated"), "{}"),
    ]

    cards = "".join(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(fmt.format(value) if value is not None else "N/A")}</div>
        </div>
        """
        for label, value, fmt in metrics
    )

    return f'<div class="metric-cards">{cards}</div>'


def build_per_class_table(summary):
    per_class = summary.get("per_class")

    if not per_class:
        return ""

    def fmt_pct(value):
        return f"{value:.1f}%" if value is not None else "N/A"

    rows = "".join(
        f"""
        <tr>
            <td>{safe_text(class_name)}</td>
            <td>{stats["tp"]}</td>
            <td>{stats["fp"]}</td>
            <td>{stats["fn"]}</td>
            <td>{fmt_pct(stats["precision"])}</td>
            <td>{fmt_pct(stats["recall"])}</td>
            <td>{fmt_pct(stats["weighted_score"])}</td>
            <td>{stats["total_pred"]}</td>
            <td>{stats["total_gt"]}</td>
            <td>{stats["count_diff"]:+d}</td>
        </tr>
        """
        for class_name, stats in per_class.items()
    )

    return f"""
    <h3>Per-class detection accuracy</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Class</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Weighted score</th>
                <th>Predicted</th>
                <th>Ground truth</th>
                <th>Count diff</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def build_counts_chart(summary, include_plotlyjs):
    per_class = summary.get("per_class") or {}

    if not per_class:
        return ""

    plot_df = pd.DataFrame(
        [{"class": name, "series": "Predicted", "count": stats["total_pred"]} for name, stats in per_class.items()]
        + [{"class": name, "series": "Ground truth", "count": stats["total_gt"]} for name, stats in per_class.items()]
    )

    fig = px.bar(
        plot_df,
        x="class",
        y="count",
        color="series",
        barmode="group",
        text="count",
        title="Predicted vs ground truth totals",
    )
    apply_chart_theme(fig)
    label_bars(fig, "%{text}")

    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def build_threshold_table(summaries_by_threshold):
    if len(summaries_by_threshold) < 2:
        return ""

    winner = best_threshold(summaries_by_threshold)

    def fmt(value):
        return f"{value:.1f}%" if value is not None else "N/A"

    rows = "".join(
        f"""
        <tr>
            <td>{safe_text(threshold)}{" (best)" if threshold == winner else ""}</td>
            <td>{fmt(summaries_by_threshold[threshold].get("map50"))}</td>
            <td>{fmt((summaries_by_threshold[threshold].get("overall") or {}).get("precision"))}</td>
            <td>{fmt((summaries_by_threshold[threshold].get("overall") or {}).get("recall"))}</td>
            <td>{fmt((summaries_by_threshold[threshold].get("overall") or {}).get("weighted_score"))}</td>
        </tr>
        """
        for threshold in sorted(summaries_by_threshold, key=float)
    )

    return f"""
    <h3>Every threshold tested</h3>
    <p class="cm-note">"Best" is the threshold with the highest weighted score above; use a different one if your use case values precision or recall differently.</p>
    <table class="results-table">
        <thead>
            <tr><th>Threshold</th><th>mAP50</th><th>Precision</th><th>Recall</th><th>Weighted score</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def generate_dashboard(model_name=None):
    """Single-run dashboard for `model_name`'s most recent test (or, when
    omitted, whichever model was tested most recently overall) - pass
    `model_name` explicitly to view an older/different model's own latest
    result instead of always the globally-last test run."""

    runs = load_runs(model_names=[model_name] if model_name else None)

    if not runs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        message = (
            f"No performance test results found yet for '{model_name}'. Run \"make performance_test MODEL_NAME={model_name}\" first."
            if model_name
            else "No performance test results found yet. Run \"make performance_test\" first."
        )
        OUTPUT_FILE.write_text(
            _empty_page("Performance Dashboard", message),
            encoding="utf-8",
        )
        return OUTPUT_FILE

    latest_run = runs[-1]
    summaries_by_threshold = latest_run.get("summary", {})
    winner = latest_run.get("best_threshold") or best_threshold(summaries_by_threshold)
    summary = summaries_by_threshold.get(winner, {})

    html_output = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Performance Dashboard</title>
        <style>{DASHBOARD_STYLES}</style>
    </head>
    <body>

        <div class="header">
            <h1>Performance Dashboard</h1>
            <div class="model-info">
                <div><strong>Model file:</strong> {safe_text(latest_run.get("model_path"))}</div>
                <div><strong>Run timestamp:</strong> {safe_text(latest_run.get("timestamp"))}</div>
                <div><strong>Threshold shown:</strong> {safe_text(winner)}</div>
            </div>
        </div>

        <section>
            {build_verdict_callout(summary)}
            {build_metric_cards(summary)}
            {build_counts_chart(summary, include_plotlyjs=True)}
            {build_per_class_table(summary)}
            {build_threshold_table(summaries_by_threshold)}
        </section>

        {build_plots_gallery(summary.get("plots_dir"))}

        {build_interpretation_guide()}

        <p class="footnote">Reports are generated locally only - see model_management/orchestrator.py to configure which metric decides promotion.</p>

    </body>
    </html>
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html_output, encoding="utf-8")

    return OUTPUT_FILE


def build_comparison_section(runs, model_names=None, include_plotlyjs=False):
    latest_by_model = {}

    for run in runs:
        model_name = run.get("model_name")

        if model_names is not None and model_name not in model_names:
            continue

        latest_by_model[model_name] = run

    if not latest_by_model:
        return ""

    try:
        active_name = get_active_model_name()
    except Exception:
        active_name = None

    rows = []

    for model_name, run in latest_by_model.items():
        summaries_by_threshold = run.get("summary", {})
        winner_threshold = run.get("best_threshold") or best_threshold(summaries_by_threshold)
        best_summary = summaries_by_threshold.get(winner_threshold, {})
        overall = best_summary.get("overall") or {}

        rows.append(
            {
                "model": model_name,
                "is_active": model_name == active_name,
                "best_threshold": winner_threshold,
                "map50": best_summary.get("map50"),
                "precision": overall.get("precision"),
                "recall": overall.get("recall"),
                "weighted_score": overall.get("weighted_score"),
                "per_class": best_summary.get("per_class") or {},
            }
        )

    def fmt_pct(value):
        return f"{value:.1f}%" if value is not None else "N/A"

    class_names = sorted({class_name for item in rows for class_name in item["per_class"]})
    class_headers = "".join(f"<th>{safe_text(class_name)} weighted score</th>" for class_name in class_names)

    def class_cells(per_class):
        return "".join(
            f"<td>{fmt_pct(per_class.get(class_name, {}).get('weighted_score'))}</td>" for class_name in class_names
        )

    table_rows = "".join(
        f"""
        <tr>
            <td>{safe_text(item['model'])}{' (active)' if item['is_active'] else ''}</td>
            <td>{safe_text(item['best_threshold'])}</td>
            <td>{fmt_pct(item['map50'])}</td>
            <td>{fmt_pct(item['precision'])}</td>
            <td>{fmt_pct(item['recall'])}</td>
            <td>{fmt_pct(item['weighted_score'])}</td>
            {class_cells(item['per_class'])}
        </tr>
        """
        for item in rows
    )

    chart_rows = [
        {"model": item["model"], "metric": metric, "value": item[key] or 0}
        for item in rows
        for metric, key in (("Precision", "precision"), ("Recall", "recall"), ("Weighted score", "weighted_score"))
    ]
    fig = px.bar(
        pd.DataFrame(chart_rows),
        x="model",
        y="value",
        color="metric",
        barmode="group",
        text="value",
        title="Precision / recall / weighted score by model (each model's own best threshold)",
    )
    apply_chart_theme(fig)
    label_bars(fig)
    chart_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    return f"""
    <section>
        <p class="cm-note">
            Each model is evaluated at its own best threshold against the same golden test dataset.
            This table only reports the numbers - see model_management/orchestrator.py to configure
            which metric (and which model) should be promoted.
        </p>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Best threshold</th>
                    <th>mAP50</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Weighted score</th>
                    {class_headers}
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        {chart_html}
    </section>
    """


def generate_comparison_dashboard(model_names=None):
    if model_names is None:
        model_names = default_downloaded_models()

    runs = load_runs(model_names=model_names)
    comparison_section = build_comparison_section(runs, model_names=model_names, include_plotlyjs=True)

    if not comparison_section:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        COMPARISON_OUTPUT_FILE.write_text(
            _empty_page("Model Comparison Dashboard", "Need results for at least 2 different models. Run \"make model_compare\" first."),
            encoding="utf-8",
        )
        return COMPARISON_OUTPUT_FILE

    html_output = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Model Comparison Dashboard</title>
        <style>{DASHBOARD_STYLES}</style>
    </head>
    <body>

        <div class="header">
            <h1>Model Comparison Dashboard</h1>
            <div class="model-info">Comparing: {safe_text(", ".join(sorted(model_names)))}</div>
        </div>

        {comparison_section}

        {build_interpretation_guide()}

        <p class="footnote">Reports are generated locally only - see model_management/orchestrator.py to configure which metric decides promotion.</p>

    </body>
    </html>
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_OUTPUT_FILE.write_text(html_output, encoding="utf-8")

    return COMPARISON_OUTPUT_FILE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the performance dashboard(s).")
    parser.add_argument("--comparison", action="store_true", help="Generate the model comparison dashboard instead of the single-run dashboard.")
    parser.add_argument("--models", nargs="+", help="Only used with --comparison: registry model names to include (default: every model currently downloaded to tmp/models).")
    parser.add_argument("--model", help="Only used without --comparison: show this model's latest result instead of whichever model was tested most recently.")
    args = parser.parse_args()

    models = [normalize_model_name(name) for name in args.models] if args.models else None
    model = normalize_model_name(args.model) if args.model else None

    path = generate_comparison_dashboard(model_names=models) if args.comparison else generate_dashboard(model_name=model)

    print(f"Dashboard generated: {path}")
