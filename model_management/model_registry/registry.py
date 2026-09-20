"""
Model registry: tracks every known YOLO model file, where it was
downloaded from, and which one is currently active.

`counter/config.py` resolves MODEL_PATH from the active model here unless
the MODEL_PATH env var overrides it. The performance test scripts use this
same registry to download and compare candidate models against it.

Each entry's `metrics_url` (optional) links to a Drive-hosted copy of the
training run's artifacts (see drive.py's upload_directory()), if one was
uploaded. Performance/evaluation reports themselves are never uploaded -
they're cached on `performance`/`performance_history` below and saved
locally only (model_management/evaluation/evaluator.py) - so "the model
link is present in the registry" is genuinely all that's needed to find a
model later.

CLI usage:
    python -m model_management.model_registry.registry list
    python -m model_management.model_registry.registry set-active yolo26_v2
    python -m model_management.model_registry.registry add <name> --file <name>.pt --url <drive-url>
    python -m model_management.model_registry.registry reset --yes
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_management.config import DEFAULT_USE_CASE
from model_management.model_registry.drive import get_storage

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"
MODEL_DIR = Path("tmp/models")

DRIVE_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)")
MARKDOWN_LINK_PATTERN = re.compile(r"^\[([^\]]+)\]\([^)]*\)$")


def extract_drive_file_id(url_or_id):
    """Accept either a raw Google Drive file ID or a full share URL."""

    match = DRIVE_ID_PATTERN.search(url_or_id)
    return match.group(1) if match else url_or_id


def normalize_model_name(name):
    """Tolerate the way model names actually get typed/pasted: a trailing
    '.pt' extension (e.g. 'yolo26.pt'), surrounding whitespace, or a chat/
    editor auto-linkified '[yolo26.pt](http://yolo26.pt)' - all resolve to
    the plain registry name 'yolo26'."""

    if not isinstance(name, str):
        return name

    name = name.strip()
    match = MARKDOWN_LINK_PATTERN.match(name)

    if match:
        name = match.group(1).strip()

    if name.lower().endswith(".pt"):
        name = name[: -len(".pt")]

    return name


def load_registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(data):
    REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _dataset_fingerprint(dataset_version):
    if dataset_version is None:
        return None

    if isinstance(dataset_version, str):
        return dataset_version

    return dataset_version.get("fingerprint")


def _ensure_test_dataset_version(data, dataset_version):
    fingerprint = _dataset_fingerprint(dataset_version)

    if not fingerprint:
        return None

    test_dataset = data.setdefault("test_dataset", {"current_version_id": None, "versions": []})
    versions = test_dataset.setdefault("versions", [])

    for version_entry in versions:
        if version_entry.get("fingerprint") == fingerprint:
            test_dataset["current_version_id"] = version_entry.get("id")
            return version_entry

    if isinstance(dataset_version, str):
        project = None
        version = None
        source_version = None
        file_counts = None
    else:
        project = dataset_version.get("project")
        version = dataset_version.get("version")
        source_version = dataset_version.get("source_version")
        file_counts = dataset_version.get("file_counts")

    version_entry = {
        "id": f"v{len(versions) + 1}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "version": version,
        "fingerprint": fingerprint,
        "file_counts": file_counts,
        "source_version": source_version,
    }

    versions.append(version_entry)
    test_dataset["current_version_id"] = version_entry["id"]

    return version_entry


def list_models():
    """Return {name: entry} for every registered model."""

    return load_registry()["models"]


def get_model(name):
    models = list_models()
    name = normalize_model_name(name)

    if name not in models:
        raise KeyError(f"Model '{name}' is not in the registry. Known models: {', '.join(models)}")

    return models[name]


def get_active_model_name():
    return load_registry()["active_model"]


def model_path(name):
    """Local path a registered model's weights are/will be stored at."""

    return MODEL_DIR / get_model(name)["file"]


def resolve_model_path(name=None):
    """Local path for `name`, or the active model when name is omitted."""

    return model_path(name or get_active_model_name())


def add_model(name, file, drive_url, metrics_url=None, use_case=DEFAULT_USE_CASE):
    data = load_registry()
    name = normalize_model_name(name)

    data["models"][name] = {
        "file": file,
        "use_case": use_case,
        "drive_url": drive_url,
        "drive_file_id": extract_drive_file_id(drive_url),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "metrics_url": metrics_url,
    }

    save_registry(data)


def get_model_use_case(name):
    return get_model(name).get("use_case", DEFAULT_USE_CASE)


def set_active_model(name):
    data = load_registry()
    name = normalize_model_name(name)

    if name not in data["models"]:
        raise KeyError(f"Model '{name}' is not in the registry. Known models: {', '.join(data['models'])}")

    data["active_model"] = name
    save_registry(data)


def get_model_performance(name, dataset_version=None):
    """Cached performance summary for `name`.

    When `dataset_version` is provided, return the cached result only if it was
    recorded against that exact golden test dataset composition.
    """

    model = get_model(name)

    if dataset_version is None:
        return model.get("performance")

    fingerprint = _dataset_fingerprint(dataset_version)

    if not fingerprint:
        return None

    for cached in reversed(model.get("performance_history", [])):
        if cached.get("dataset_fingerprint") == fingerprint:
            return cached

    performance = model.get("performance")

    if performance and performance.get("dataset_fingerprint") == fingerprint:
        return performance

    return None


def _round_floats(value, digits=3):
    """Recursively round floats so near-identical reruns (float noise from
    a non-deterministic val() pass) still compare as unchanged."""

    if isinstance(value, float):
        return round(value, digits)

    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}

    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]

    return value


def _metrics_unchanged(previous, candidate):
    """True if `candidate`'s actual scores match `previous`'s - ignoring
    bookkeeping fields like tested_at/results_path/plots_dir that always
    differ between runs even when nothing about the model's accuracy did."""

    if previous is None:
        return False

    compare_keys = ("overall", "per_class", "map50", "map50_95", "images_evaluated", "threshold")

    return all(_round_floats(previous.get(key)) == _round_floats(candidate.get(key)) for key in compare_keys)


def set_model_performance(name, performance, dataset_version=None):
    """Cache a golden-dataset performance summary for `name` in the registry
    so compare_models.py can skip re-running the test next time, as long as
    the golden dataset composition hasn't changed since.

    History is only appended/updated when the scores actually differ from
    the last recorded run against the same dataset version (this can happen
    when the golden dataset's images/labels changed but its version wasn't
    bumped) - re-testing an unchanged model against an unchanged dataset
    just leaves the existing history entry alone instead of logging noise.
    """

    data = load_registry()
    name = normalize_model_name(name)

    if name not in data["models"]:
        raise KeyError(f"Model '{name}' is not in the registry. Known models: {', '.join(data['models'])}")

    model_entry = data["models"][name]
    performance_record = dict(performance)
    version_entry = _ensure_test_dataset_version(data, dataset_version)

    if version_entry is not None:
        # Only the id/fingerprint are kept (not the full dataset_version dict,
        # which carries this machine's absolute path) - test_dataset.versions[]
        # above is already the single source of truth for a version's details.
        performance_record["dataset_version_id"] = version_entry["id"]
        performance_record["dataset_fingerprint"] = version_entry["fingerprint"]
        performance_record.pop("golden_dataset_version", None)

        history = model_entry.setdefault("performance_history", [])
        previous_entry = next(
            (entry for entry in history if entry.get("dataset_version_id") == version_entry["id"]),
            None,
        )

        if _metrics_unchanged(previous_entry, performance_record):
            save_registry(data)
            return

        history = [entry for entry in history if entry.get("dataset_version_id") != version_entry["id"]]
        history.append(performance_record)
        model_entry["performance_history"] = history

    model_entry["performance"] = performance_record
    save_registry(data)


def is_downloaded(name):
    path = model_path(name)
    return path.exists() and path.stat().st_size > 0


def reset_test_dataset():
    """Clear dataset-version history and every model's cached performance, for
    a fresh start (e.g. switching to an unrelated golden dataset/use case).
    Registered models (`models{}`/`active_model`) are left untouched - only
    the evaluation history tied to the old golden dataset is cleared, since
    the next evaluation run will recreate `test_dataset` from v1 onward."""

    data = load_registry()

    data["test_dataset"] = {"current_version_id": None, "versions": []}

    for model_entry in data["models"].values():
        model_entry["performance"] = None
        model_entry["performance_history"] = []

    save_registry(data)


def ensure_downloaded(name):
    """Download the model's weights via the registry's WeightsStorage backend
    (see model_registry/ports.py) if not already present."""

    name = normalize_model_name(name)
    entry = get_model(name)
    destination = model_path(name)

    if is_downloaded(name):
        return destination

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model '{name}' ({entry['file']})...")

    get_storage().download(entry["drive_file_id"], destination)

    verify = subprocess.run(
        [sys.executable, "-c", f"from ultralytics import YOLO; YOLO('{destination}')"],
        capture_output=True,
    )

    if verify.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file for model '{name}' is not a valid YOLO model.")

    print(f"Model '{name}' downloaded to {destination}")

    return destination


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage the YOLO model registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List every registered model.")

    set_active_parser = subparsers.add_parser("set-active", help="Set the active model.")
    set_active_parser.add_argument("name")

    add_parser = subparsers.add_parser("add", help="Register a new model.")
    add_parser.add_argument("name")
    add_parser.add_argument("--file", required=True, help="Filename under tmp/models/")
    add_parser.add_argument("--url", required=True, help="Google Drive share URL")
    add_parser.add_argument("--metrics-url", default=None, help="Drive link to the training report/dashboard")
    add_parser.add_argument("--use-case", default=DEFAULT_USE_CASE, help=f"Use case this model belongs to (default: {DEFAULT_USE_CASE}). See model_management/config.py.")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Clear dataset-version history and every model's cached performance (keeps registered models).",
    )
    reset_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    args = parser.parse_args()

    if args.command == "list":
        data = load_registry()

        for name, entry in data["models"].items():
            marker = " (active)" if name == data["active_model"] else ""
            downloaded = "downloaded" if is_downloaded(name) else "not downloaded"
            use_case = f" | use case: {entry.get('use_case', DEFAULT_USE_CASE)}"
            metrics = f" | metrics: {entry['metrics_url']}" if entry.get("metrics_url") else ""
            performance = entry.get("performance")
            score = (
                f" | weighted score: {performance['overall']['weighted_score']:.1f}% (threshold {performance['threshold']})"
                if performance
                else ""
            )
            print(f"{name}{marker}: {entry['file']} [{downloaded}]{use_case} <- {entry['drive_url']}{metrics}{score}")

    elif args.command == "set-active":
        set_active_model(args.name)
        print(f"Active model set to '{normalize_model_name(args.name)}'.")

    elif args.command == "add":
        add_model(args.name, args.file, args.url, args.metrics_url, use_case=args.use_case)
        print(f"Added model '{normalize_model_name(args.name)}' ({args.file}), use case '{args.use_case}'.")

    elif args.command == "reset":
        if not args.yes and input(
            "This clears test_dataset version history and every model's cached "
            "performance in registry.json. Continue? [y/N] "
        ).strip().lower() != "y":
            print("Aborted.")
            return

        reset_test_dataset()
        print("Registry reset: cleared test_dataset version history and all cached performance.")


if __name__ == "__main__":
    _main()
