from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clearml_integration import call_clearml_serving


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a prediction request to ClearML Serving.")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/serve/predictive_maintenance",
    )
    parser.add_argument("--type", default="L", choices=["L", "M", "H"])
    parser.add_argument("--air-temp", type=float, default=300.0)
    parser.add_argument("--process-temp", type=float, default=310.0)
    parser.add_argument("--speed", type=int, default=1500)
    parser.add_argument("--torque", type=float, default=40.0)
    parser.add_argument("--tool-wear", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = {
        "Type": args.type,
        "Air temperature [K]": args.air_temp,
        "Process temperature [K]": args.process_temp,
        "Rotational speed [rpm]": args.speed,
        "Torque [Nm]": args.torque,
        "Tool wear [min]": args.tool_wear,
    }
    response = call_clearml_serving(args.endpoint, values)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
