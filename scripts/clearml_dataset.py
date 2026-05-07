from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clearml_integration import CLEARML_PROJECT, create_clearml_dataset
from maintenance_core import DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and upload a ClearML Dataset version.")
    parser.add_argument("--project", default=CLEARML_PROJECT)
    parser.add_argument("--name", default="AI4I 2020 Predictive Maintenance")
    parser.add_argument("--data", default=str(DATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_id = create_clearml_dataset(
        args.data,
        dataset_name=args.name,
        dataset_project=args.project,
    )
    print(dataset_id)


if __name__ == "__main__":
    main()
