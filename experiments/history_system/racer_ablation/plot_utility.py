"""Recovery utility figure; reads only utility_curve.py output.

usage: python -m racer_ablation.plot_utility UTILITY_JSON OUT_PDF
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"full": ("#0072B2", "-", "Full detector"),
         "light": ("#E69F00", "--", "Light detector (NLL + STOP + parse)"),
         "random_expected": ("#999999", ":", "Uniform random (expected)")}


def main(argv):
    data = json.load(open(argv[0], encoding="utf-8"))
    curves, bands = data["curves"], data["bands"]
    x = curves["x"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, (color, style, label) in STYLE.items():
        if name not in curves:
            continue
        ax.fill_between(bands["grid"], [100 * v for v in bands[name]["lo"]],
                        [100 * v for v in bands[name]["hi"]], color=color, alpha=0.12, linewidth=0)
        ax.plot(x, [100 * value for value in curves[name]], color=color, linestyle=style,
                linewidth=1.4, label=label)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Extra generations per state ($k/N$)")
    ax.set_ylabel("Net current-turn success (pp)")
    ax.set_xlim(0, x[-1] if x else 1)
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    ax.set_title(f"N={data['n']} states, recoverable pool {data['pool']} (preliminary, n=1)",
                 fontsize=7)
    fig.tight_layout()
    fig.savefig(argv[1])
    print("wrote", argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
