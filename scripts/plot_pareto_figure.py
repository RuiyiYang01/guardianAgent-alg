"""
Generate Pareto scatter plot for the paper.

Two side-by-side panels (TAB / SynthPAI). X axis = utility, Y axis = privacy.
Each method is a point; ours is a star with error bars (4-seed std).
The Pareto frontier is drawn as a connecting line through frontier points.

Output: paper/paper-algorithm-emnlp/latex/figure/pareto.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG_OUT = ROOT.parent / "paper" / "paper-algorithm-emnlp" / "latex" / "figure" / "pareto.pdf"
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)


def load_4seed(corpus: str):
    p = RES / "variance_main_n200_5level.json"
    data = json.loads(p.read_text())
    out = {}
    cfgs = data["results"][corpus]
    for name, m in cfgs.items():
        out[name] = {
            "priv": m["avg_privacy"]["mean"],
            "priv_std": m["avg_privacy"]["std"],
            "util": m["avg_utility"]["mean"],
            "util_std": m["avg_utility"]["std"],
        }
    return out


def short_name(full: str) -> str:
    if full.startswith("D. Ours"):
        return "Ours (full)"
    parts = full.split(". ", 1)
    if len(parts) < 2:
        return full
    rest = parts[1]
    # Take first word/phrase
    short = rest.split(" (")[0].split(":")[0]
    short = short.replace("Ours", "Ours").strip()
    return short


def is_ours(name: str) -> bool:
    return name.startswith(("A. Ours", "B. Ours", "C. Ours", "D. Ours"))


def pareto_frontier(points):
    """points: list of (util, priv, name). Return frontier sorted by util ascending."""
    sorted_pts = sorted(points, key=lambda p: (-p[1], -p[0]))
    frontier = []
    max_util = -1.0
    for p in sorted_pts:
        if p[0] > max_util:
            frontier.append(p)
            max_util = p[0]
    return sorted(frontier, key=lambda p: p[0])


def plot_panel(ax, data, title, exclude=None):
    exclude = exclude or set()
    points_all = []
    for name, m in data.items():
        if name in exclude:
            continue
        points_all.append((m["util"], m["priv"], name, m["util_std"], m["priv_std"]))

    # Pareto frontier on (priv, util) — maximise both
    frontier_pts = pareto_frontier([(p[0], p[1], p[2]) for p in points_all])

    # Draw frontier line
    fx = [p[0] for p in frontier_pts]
    fy = [p[1] for p in frontier_pts]
    ax.plot(fx, fy, color="#999", linestyle="--", linewidth=1, zorder=1, label="Pareto frontier")

    # Plot baselines (non-D)
    for util, priv, name, us, ps in points_all:
        if name.startswith("D. Ours"):
            continue
        is_o = is_ours(name)
        is_frontier = (util, priv, name) in frontier_pts
        color = "#1f77b4" if is_o else ("#d62728" if is_frontier else "#888")
        marker = "s" if is_o else ("^" if is_frontier else "o")
        size = 50 if is_frontier else 35
        ax.scatter([util], [priv], s=size, marker=marker, color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        # Label
        label = short_name(name)
        # Offset labels to avoid clutter
        dx, dy = 0.005, 0.005
        ax.annotate(label, (util, priv), xytext=(util + dx, priv + dy),
                    fontsize=7, color="#333", zorder=4)

    # Plot D with error bars
    if "D. Ours: LLM-anon+guesser" in data:
        m = data["D. Ours: LLM-anon+guesser"]
        ax.errorbar([m["util"]], [m["priv"]],
                    xerr=[m["util_std"]], yerr=[m["priv_std"]],
                    fmt="*", markersize=18, color="#2ca02c",
                    markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.2, capsize=3, capthick=1, zorder=5,
                    label="Ours (full, L1-L5)")
        ax.annotate("Ours (full)", (m["util"], m["priv"]),
                    xytext=(m["util"] + 0.012, m["priv"] - 0.01),
                    fontsize=8, fontweight="bold", color="#2ca02c", zorder=6)

    ax.set_xlabel("Utility ↑")
    ax.set_ylabel("Privacy ↑")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlim(left=0.4)
    ax.set_ylim(bottom=0.1)


def main():
    tab = load_4seed("tab")
    synth = load_4seed("staab-synth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Exclude only configs A and B from baseline comparison panels (they are our own ablations
    # the main message is wrt published baselines + our flagship D)
    exclude = {"A. Ours: NER-only", "B. Ours: NER+LLM-guesser"}
    plot_panel(ax1, tab, "TAB (legal)", exclude=exclude)
    plot_panel(ax2, synth, "SynthPAI (Reddit)", exclude=exclude)

    # Annotate Staab artefact on SynthPAI
    if "I. Staab (ICLR 2025) [upstream]" in synth:
        m = synth["I. Staab (ICLR 2025) [upstream]"]
        ax2.annotate("(broken outputs)", (m["util"], m["priv"]),
                     xytext=(m["util"] + 0.02, m["priv"] - 0.025),
                     fontsize=6, style="italic", color="#a00")

    fig.tight_layout()
    fig.savefig(FIG_OUT, bbox_inches="tight")
    print(f"Saved: {FIG_OUT}")
    print(f"Size: {FIG_OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
