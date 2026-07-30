from pathlib import Path

import matplotlib.pyplot as plt


STEPS = [8, 4, 2, 1]
RESULTS = {
    "DSM = 0.1": {
        "success_rate": [97.708, 96.250, 95.417, 6.458],
        "mode_entropy": [0.88449, 0.87965, 0.81807, 0.61855],
        "color": "#1f77b4",
        "marker": "o",
    },
    "DSM = 0": {
        "success_rate": [97.708, 96.042, 94.792, 0.000],
        "mode_entropy": [0.88449, 0.89764, 0.84380, 0.00000],
        "color": "#d62728",
        "marker": "s",
    },
}


def plot_metric(key, ylabel, filename, ylim, value_format):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for label, values in RESULTS.items():
        ys = values[key]
        ax.plot(
            STEPS,
            ys,
            linewidth=2.4,
            marker=values["marker"],
            markersize=7,
            color=values["color"],
            label=label,
        )
        for x, y in zip(STEPS, ys):
            ax.annotate(
                value_format.format(y),
                (x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=values["color"],
            )

    ax.set_xticks(STEPS)
    ax.set_xlim(8.5, 0.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("DDIM inference steps (progressive distillation: 8 → 4 → 2 → 1)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(
        "success_rate",
        "Closed-loop success rate (%)",
        output_dir / "distillation_success_rate_vs_steps.png",
        (-3, 105),
        "{:.1f}%",
    )
    plot_metric(
        "mode_entropy",
        "Normalized mode entropy",
        output_dir / "distillation_mode_entropy_vs_steps.png",
        (-0.03, 1.0),
        "{:.3f}",
    )


if __name__ == "__main__":
    main()
