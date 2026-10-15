"""Analysis entry point using NumPy from the isolated AISBench environment."""

import argparse
import json
from pathlib import Path
from . import metrics, observation
from .util import read_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        result = metrics.summarize(args.path)
        print(
            json.dumps(
                {"complete": result["complete"], "valid_runs": result["valid_runs"]}
            )
        )
        return
    arm = args.path
    config = read_json(arm.parent.parent / "config.json")
    manifest = read_json(arm.parent.parent / "inputs/manifest.json")
    performance, samples = metrics.extract(arm, config, manifest)
    checks = dict(performance["checks"])
    execution = read_json(arm / "aisbench/execution.json")
    checks["aisbench_exit_success"] = execution["returncode"] == 0
    result = {**read_json(arm / "arm.json"), "performance": performance}
    if config["observation"]["enabled"]:
        observed = observation.analyze(arm, config, manifest, samples)
        checks.update(observed["checks"])
        result["observation"] = observed
    else:
        result["observation"] = None
    result.update(checks=checks, passed=all(checks.values()))
    save_json(arm / "analysis.json", result)
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
