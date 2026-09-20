"""
Shared "which model wins" logic used by model_management/orchestrator.py and
model_management/evaluation/dashboard/compare_models.py, so the promotion
rule lives in exactly one place instead of being hard-coded differently in
each caller (see the module docstring in orchestrator.py for why this
matters: the dashboards only report metrics, they never decide a winner).

A metric is any key under a performance summary's "overall" dict - currently
"precision", "recall", or "weighted_score" (F1). There is no built-in
default preference for one class or metric over another; callers choose.
"""

METRICS = ("precision", "recall", "weighted_score")
DEFAULT_METRIC = "weighted_score"


def metric_value(summary, metric=DEFAULT_METRIC):
    """Extract `metric` from a performance summary dict (either the shape
    evaluator.run_for_model()'s `summaries[threshold]` uses, or the flat
    shape cached on a registry entry's `performance`/`performance_history`)."""

    if summary is None:
        return None

    overall = summary.get("overall") or {}

    return overall.get(metric)


def candidate_beats_active(candidate_summary, active_summary, metric=DEFAULT_METRIC):
    """True if the candidate's `metric` is strictly better than the active
    model's, using the same summary shape as metric_value()."""

    candidate_score = metric_value(candidate_summary, metric)
    active_score = metric_value(active_summary, metric)

    if candidate_score is None:
        return False

    if active_score is None:
        return True

    return candidate_score > active_score


def rank_models(summaries_by_model, metric=DEFAULT_METRIC):
    """Return model names sorted best-to-worst by `metric` (None scores last)."""

    def sort_key(name):
        value = metric_value(summaries_by_model[name], metric)
        return (value is None, -(value or 0))

    return sorted(summaries_by_model, key=sort_key)
