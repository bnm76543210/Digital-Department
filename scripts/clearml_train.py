from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clearml_integration import CLEARML_PROJECT, log_training_report
from maintenance_core import DATA_PATH, read_dataset, result_table, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models and log the experiment to ClearML.")
    parser.add_argument("--project", default=CLEARML_PROJECT)
    parser.add_argument("--task", default="CLI training run")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--optimize-rf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_dataset(args.data)
    report = train_models(data, cv_splits=args.cv, optimize_random_forest=args.optimize_rf)
    print(result_table(report).to_string(index=False))
    clearml_result = log_training_report(
        report,
        data=data,
        project_name=args.project,
        task_name=args.task,
    )
    print(clearml_result)


if __name__ == "__main__":
    main()
