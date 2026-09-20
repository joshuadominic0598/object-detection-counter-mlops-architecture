#!/usr/bin/env python3
"""
Downloads YOLO model weights from Google Drive using the model registry
(model_management/model_registry/registry.json). Existing files are skipped.

Usage:
    python scripts/download_model.py                       # download every registered model
    python scripts/download_model.py --active              # download only the active model
    python scripts/download_model.py --models NAME [NAME ...]  # download specific models (".pt" suffixes/pasted links are tolerated)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_management.model_registry.registry import ensure_downloaded, get_active_model_name, list_models, normalize_model_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Download every registered model (default when no other option is given)")
    parser.add_argument("--active", action="store_true", help="Download only the active model")
    parser.add_argument("--models", nargs="+", help="Download specific registered models by name")
    args = parser.parse_args()

    if args.models:
        names = [normalize_model_name(name) for name in args.models]
    elif args.active:
        names = [get_active_model_name()]
    else:
        names = list(list_models())

    for name in names:
        path = ensure_downloaded(name)
        print(f"'{name}' ready at {path}")


if __name__ == "__main__":
    main()
