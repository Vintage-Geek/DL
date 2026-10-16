"""Compare already-saved evaluation reports; does not select a test winner."""

from pathlib import Path

from src.visualization.comparison import collect_experiments, write_comparison


def run_comparison(outputs_dir: str | Path = "outputs") -> tuple[Path, Path]:
    rows = collect_experiments(outputs_dir)
    if not rows:
        raise SystemExit(f"No evaluated experiments found under {Path(outputs_dir) / 'experiments'}.")
    csv_path, plot_path = write_comparison(outputs_dir, rows)
    print(f"Saved {len(rows)} experiment rows to {csv_path} and {plot_path}")
    return csv_path, plot_path


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()