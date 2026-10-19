"""Recompute cloud diagnostics and make report figures/tables from saved data."""

from pathlib import Path
import json
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cirq_sic.noise import tomography_cloud_metrics, compare_state_spaces
from cirq_sic.measurements import sic_povm
from cirq_sic.triples import triple_products
from research.noise_state_space.run_study import clean

base = Path(__file__).parent
out = base / "results"
report = base / "report"
rows = json.loads((out / "summary.json").read_text())
reference = triple_products(sic_povm(2).effects * 2)
for row in rows:
    a = np.load(out / row["data"])
    if row["paired_tomography"]:
        row["tomography"] = clean(
            tomography_cloud_metrics(a["exact_cloud"], a["ideal_cloud"], a["tensor"], 2)
        )
        row["sampled_tomography"] = clean(
            tomography_cloud_metrics(
                a["sampled_cloud"], a["ideal_cloud"], a["sampled_tensor"], 2
            )
        )
(out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 160,
    }
)
colors = ["#287c8e", "#d77b24", "#8155a0"]
names = ["depolarizing", "bit_flip", "amplitude_damping"]
labels = ["Depolarization", "Bit flips", "Amplitude damping"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
for ax, name, label, color in zip(axes, names, labels, colors):
    selected = [
        r for r in rows if r["name"] == name and r["placement"] == "preparation"
    ]
    x = [0] + [r["rate"] for r in selected]
    ax.plot(
        x,
        [1] + [r["geometry"].get("relative_volume", np.nan) for r in selected],
        "o-",
        color=color,
        label="Triple-derived body",
    )
    ax.plot(
        x,
        [1] + [r["tomography"]["affine_ball_volume_ratio"] for r in selected],
        "s--",
        color="#334155",
        label="Tomography affine image",
    )
    for r in selected:
        if r["geometry"]["status"] != "ok":
            ax.text(r["rate"], 0.08, "invalid\ncenter", ha="center", fontsize=8)
    ax.set(
        title=label,
        xlabel="Channel parameter",
        ylabel="Volume / true volume",
        ylim=(0, 1.06),
    )
axes[0].legend(fontsize=8, loc="lower left")
fig.savefig(report / "volumes.pdf")
fig.savefig(report / "volumes.png")
plt.close(fig)

basis = null_space(np.ones((1, 4)))
theta = np.linspace(0, 2 * np.pi, 361)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
for ax, name, label, color in zip(axes, names, labels, colors):
    r = next(
        r
        for r in rows
        if r["name"] == name and r["placement"] == "preparation" and r["rate"] == 0.05
    )
    a = np.load(out / r["data"])
    g = compare_state_spaces(a["tensor"], reference, 2, directions=12000, seed=123)
    points = (g["directions"] * g["radii"][:, None]) @ basis
    cloud = a["exact_cloud"] @ basis
    for points2, c, style, lab in [
        (points[:, :2], color, "-", "Triple projection"),
        (cloud[:, :2], "#334155", "--", "Tomography cloud hull"),
    ]:
        h = ConvexHull(points2)
        v = np.r_[h.vertices, h.vertices[0]]
        ax.plot(points2[v, 0], points2[v, 1], style, color=c, label=lab)
    ax.plot(
        np.cos(theta) / np.sqrt(12),
        np.sin(theta) / np.sqrt(12),
        ":",
        color="gray",
        label="True ball projection",
    )
    ax.set(
        title=label + " (0.05)",
        xlabel="Probability coordinate 1",
        ylabel="Probability coordinate 2",
        aspect="equal",
        xlim=(-0.32, 0.32),
        ylim=(-0.32, 0.32),
    )
axes[0].legend(fontsize=7, loc="lower left")
fig.savefig(report / "projections.pdf")
fig.savefig(report / "projections.png")
plt.close(fig)

selected = [
    r
    for r in rows
    if (r["placement"] in ("preparation", "moment_all") and r["rate"] == 0.05)
    or r["placement"] == "none"
    or r["name"] == "coherent_z"
    or (r["placement"] == "readout" and r["rate"] == 0.05)
]


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


lines = [
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Channel & Placement & $V_T/V_0$ & $V_{\rm tomo}/V_0$ & TV & $\epsilon_{\rm perm}$ \\",
    r"\midrule",
]
for r in selected:
    t = r.get("tomography", {})
    lines.append(
        f"{r['name'].replace('_',' ')} & {r['placement'].replace('_',' ')} & {fmt(r['geometry'].get('relative_volume'))} & {fmt(t.get('affine_ball_volume_ratio'))} & {fmt(t.get('paired_tv_mean'))} & {r['diagnostics']['permutation_rms']:.4f}"
        + r" \\"
    )
lines.extend([r"\bottomrule", r"\end{tabular}"])
(report / "table.tex").write_text("\n".join(lines) + "\n")
print("Generated figures and table; refreshed affine diagnostics.")
for r in rows:
    if r in selected:
        print(
            r["name"],
            r["placement"],
            r["geometry"].get("relative_volume"),
            r.get("tomography", {}).get("affine_ball_volume_ratio"),
        )
