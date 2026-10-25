"""Standalone matplotlib figure from audited comparison JSON; no learner imports."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    content = args.report.read_bytes()
    report = json.loads(content)
    choices = [
        ("quiet160-f0-l3.5-b1.5", "Quiet160", "#516573", (-10, 10)),
        ("quiet128-f0-l3.5-b1.5", "Quiet128", "#9babb6", (8, -17)),
        ("pcen160-f0-l2.5-b1.5", "PCEN selected", "#bd640c", (8, 10)),
        ("mcra128-f0-l3.5-b1.5", "MCRA diagnostic", "#a34146", (8, -17)),
        ("adaptive_gain", "Adaptive gain", "#167d79", (8, 10)),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 5.1), gridspec_kw={"width_ratios": [1.6, 1]})
    for index, (name, label, color, offset) in enumerate(choices):
        metrics = report["modes"][name]["metrics"]
        x, y, timing = (metrics[key] for key in ("control_admissions", "strict_episode_exclusion", "observe_ms"))
        error = lambda metric, scale: [[scale*(metric["mean"]-metric["bootstrap_95"][0])],
                                        [scale*(metric["bootstrap_95"][1]-metric["mean"])]]
        left.errorbar(x["mean"], 100*y["mean"], xerr=error(x, 1), yerr=error(y, 100),
                      fmt="o", color=color, markersize=6, capsize=3, linewidth=1.3)
        left.annotate(label, (x["mean"], 100*y["mean"]), xytext=offset, textcoords="offset points",
                      ha="right" if offset[0] < 0 else "left", color=color, fontsize=9)
        right.errorbar(timing["mean"], index, xerr=error(timing, 1), fmt="o", color=color,
                       markersize=6, capsize=3, linewidth=1.3)
    left.set(xlabel="Background commitments per 210 s control stream", ylabel="Strict source-exclusion repeat identity (%)",
             title="Foreground retention and background learning", ylim=(0, 100), xlim=(0, None))
    right.set(yticks=list(range(len(choices))), yticklabels=[row[1] for row in choices],
              xlabel="Mean observe time per 1 s audio (ms)", title="Warmed Python inference", xlim=(0, None))
    right.invert_yaxis()
    right.xaxis.set_major_locator(MaxNLocator(5))
    for axis in (left, right):
        axis.grid(axis="x", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Recorded frontend comparison: held-out source recordings", x=.05, ha="left", fontsize=14, weight="bold")
    fig.text(.05, .025, "10 mixer seeds; five fixed foreground recordings. Bars: marginal 95% seed-bootstrap intervals.\n"
             "Desktop timing is not CPU-isolated. MCRA did not meet development retention requirements.", fontsize=8.5, color="#4c4c4c")
    fig.subplots_adjust(left=.08, right=.97, bottom=.21, top=.83, wspace=.65)
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("comparison."+suffix), dpi=180, facecolor="white")
    plt.close(fig)
    inputs = {"input/report.json": content}
    source_name = "python/src/chuff_mpu/sim/plot_published.py"
    sources = {source_name: Path(__file__).read_bytes()}
    for name, data in {**inputs, **{"source/"+name: data for name, data in sources.items()}}.items():
        destination = args.output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    manifest = dict(schema="chuff.recorded.published-figure.v1", matplotlib=matplotlib.__version__,
        input_sha256={name: digest(data) for name, data in inputs.items()},
        source_sha256={name: digest(data) for name, data in sources.items()},
        output_sha256={"comparison."+suffix: digest((args.output/("comparison."+suffix)).read_bytes())
                       for suffix in ("png", "svg")})
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
