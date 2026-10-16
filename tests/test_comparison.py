import json

from scripts.compare import run_comparison


def test_comparison_uses_configured_output_root(tmp_path):
    run_dir = tmp_path / "experiments" / "synthetic" / "P0" / "seed-1"
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"dataset": {"name": "synthetic"}, "experiment": {"preprocessing": "P0", "seed": 1}}),
        encoding="utf-8",
    )
    (evaluation_dir / "metrics.json").write_text(
        json.dumps({"accuracy": 0.5, "macro_precision": 0.5, "macro_recall": 0.5, "macro_f1": 0.5, "inference_seconds": 0.1, "sample_count": 2}),
        encoding="utf-8",
    )

    csv_path, plot_path = run_comparison(tmp_path)

    assert csv_path == tmp_path / "comparisons" / "comparison.csv"
    assert plot_path == tmp_path / "comparisons" / "macro_f1_comparison.png"
    assert csv_path.is_file()
    assert plot_path.is_file()