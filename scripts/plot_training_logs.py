from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
FIGURE_DIR = ROOT / "figures"


def plot_run(csv_name: str, output_name: str, title: str) -> None:
    log_path = LOG_DIR / csv_name
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    data = pd.read_csv(log_path)
    required = {"step", "total_loss", "kd_loss", "ce_loss"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{log_path} is missing columns: {sorted(missing)}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    ax.plot(data["step"], data["total_loss"], marker="o", label="Total")
    ax.plot(data["step"], data["kd_loss"], marker="s", label="KD")
    ax.plot(data["step"], data["ce_loss"], marker="^", label="CE")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=False)

    output_path = FIGURE_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    plot_run(
        "B_20K_training.csv",
        "loss_B_20K_full.pdf",
        "20K configuration, 62.09M parameters",
    )
    plot_run(
        "C_10K_training_partial.csv",
        "loss_C_10K_partial.pdf",
        "10K configuration, 64.78M parameters (partial)",
    )


if __name__ == "__main__":
    main()
