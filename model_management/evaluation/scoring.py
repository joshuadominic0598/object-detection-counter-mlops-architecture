"""
Scoring logic for evaluating a YOLO model against the golden test dataset
(model_management/evaluation/golden_dataset). Metrics are derived from
Ultralytics' own box-matched confusion matrix (real IoU-based TP/FP/FN), not
aggregate per-image count differences, so precision/recall reflect actual
detections rather than lucky count matches.

A prediction only counts as a true positive when it overlaps a ground-truth
box by at least IOU_MATCH_THRESHOLD - i.e. the box has to be "close enough",
not just the right class anywhere in the image. That is what makes count
diffs a sanity check rather than the actual accuracy measure: a missed car
and a spurious car can cancel out to a count diff of 0 while both being real
errors, which the per-class TP/FP/FN below would still catch.

This module is intentionally domain-agnostic - it has no notion of which
class "matters more" (e.g. no built-in weighting favoring one class over
another). Every class is scored the same way; which metric a promotion
decision should be based on is a choice made by the caller (see
model_management/promotion.py), not baked into the scoring itself.

Used by model_management/evaluation/evaluator.py (computed right after a
model.val() run) and model_management/evaluation/dashboard/generate_performance_dashboard.py
(recomputing display data from the summaries already stored in
results/model_management/evaluation/*.json).
"""

# Ultralytics' confusion matrix only counts a prediction as a true positive
# for a ground-truth box when their IoU is at least this - i.e. "close
# enough" overlap, not just any prediction of the right class anywhere in
# the image. Surfaced in dashboards so the matching criteria is explicit
# rather than an implicit assumption.
IOU_MATCH_THRESHOLD = 0.45


def f_beta_score(precision, recall, beta=1.0):
    if precision is None or recall is None or (precision + recall) == 0:
        return 0.0

    numerator = (1 + beta ** 2) * precision * recall
    denominator = (beta ** 2 * precision) + recall

    return numerator / denominator if denominator > 0 else 0.0


def class_metrics_from_confusion(matrix, class_index, class_name):
    """
    matrix: Ultralytics ConfusionMatrix.matrix, shape (nc+1, nc+1). Rows are
    the predicted class (last row = background, i.e. an unmatched
    prediction), columns are the ground-truth class (last column =
    background, i.e. a ground-truth object with no matching prediction).
    """

    tp = float(matrix[class_index, class_index])
    fp = float(matrix[class_index, :].sum()) - tp
    fn = float(matrix[:, class_index].sum()) - tp

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    # F1 (beta=1): precision and recall weighted equally. This is the
    # neutral default "weighted_score" - callers that care about a
    # different tradeoff (e.g. favoring recall) can recompute with a
    # different beta rather than have one baked in here.
    f1 = f_beta_score(precision, recall, beta=1.0) if precision is not None and recall is not None else None

    total_pred = int(tp + fp)
    total_gt = int(tp + fn)

    return {
        "class_name": class_name,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision * 100 if precision is not None else None,
        "recall": recall * 100 if recall is not None else None,
        "f1": f1 * 100 if f1 is not None else None,
        "weighted_score": f1 * 100 if f1 is not None else None,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "count_diff": total_pred - total_gt,
        "count_abs_diff": abs(total_pred - total_gt),
    }


def overall_metrics(per_class):
    """Macro-average precision/recall/f1/weighted_score across classes."""

    scored = [class_summary for class_summary in per_class.values() if class_summary["precision"] is not None]

    if not scored:
        return {"precision": None, "recall": None, "f1": None, "weighted_score": None}

    def avg(key):
        return sum(class_summary[key] for class_summary in scored) / len(scored)

    return {
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "weighted_score": avg("weighted_score"),
    }


def summarize_validation(
    threshold, class_names, confusion_matrix, map50, map50_95, images_evaluated, plots_dir=None
):
    """
    Build the per-threshold summary dict from a model.val() run's outputs.

    class_names: {index: name}, confusion_matrix: the raw (nc+1, nc+1)
    matrix from metrics.confusion_matrix.matrix. plots_dir (optional): repo-root-relative
    path to the confusion matrix / PR / F1 curve PNGs Ultralytics saved for this run.
    """

    per_class = {
        name: class_metrics_from_confusion(confusion_matrix, index, name)
        for index, name in class_names.items()
    }

    return {
        "threshold": threshold,
        "images_evaluated": images_evaluated,
        "map50": map50 * 100 if map50 is not None else None,
        "map50_95": map50_95 * 100 if map50_95 is not None else None,
        "per_class": per_class,
        "overall": overall_metrics(per_class),
        "plots_dir": plots_dir,
    }


def verdict_tier(score):
    """Classify a 0-100 score into a plain-language tier + CSS class so
    dashboards can show a qualitative label alongside the raw number."""

    if score is None:
        return "Unknown", "tier-warn"

    if score >= 80:
        return "Strong", "tier-ok"

    if score >= 60:
        return "Moderate", "tier-warn"

    return "Weak", "tier-bad"


def best_threshold(summaries_by_threshold, metric="weighted_score"):
    """Threshold with the highest overall (macro-averaged) value for `metric`."""

    scored = {
        threshold: summary["overall"].get(metric)
        for threshold, summary in summaries_by_threshold.items()
        if summary.get("overall", {}).get(metric) is not None
    }

    if not scored:
        return None

    return max(scored, key=scored.get)
