"""Sequentially train and evaluate configured preprocessing arms."""

from __future__ import annotations

import argparse

from scripts.common import load_config
from scripts.train import run
from scripts.evaluate import evaluate_run
from scripts.compare import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    arms = config["experiment"].get("preprocessing_arms", ["P0"])
    for arm in arms:
        run_dir = run(config, arm)
        evaluate_run(run_dir)
    run_comparison(config.get("outputs_dir", "outputs"))


if __name__ == "__main__":
    main()