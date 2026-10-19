"""Compiled-circuit sensitivity and equal-total-shot tomography checks."""

from pathlib import Path
import sys, json
import cirq
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cirq_sic.noise import *
from cirq_sic.measurements import sic_povm
from cirq_sic.triples import triple_products
from research.noise_state_space.run_study import clean

out = Path(__file__).parent / "results"
ref = triple_products(sic_povm(2).effects * 2)
compiled = []
for rate in [0.001, 0.01]:
    depths = []

    def transform(c):
        cc = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset())
        depths.append((len(c), len(cc)))
        return cc

    r = simulate_triple_tensor(
        noise=LocalMomentNoise(cirq.depolarize(rate)),
        ordered=False,
        circuit_transform=transform,
    )
    compiled.append(
        clean(
            {
                "rate": rate,
                "ordered": False,
                "mean_moments_before_after": np.mean(depths, axis=0),
                "geometry": compare_state_spaces(r.tensor, ref, 2),
                "diagnostics": triple_tensor_diagnostics(r.tensor, 2, reference=ref),
            }
        )
    )
    np.savez_compressed(
        out / f"compiled_depolarize_{rate}.npz",
        triplets=r.triplets,
        estimates=r.estimates,
        tensor=r.tensor,
    )
(out / "compiled_summary.json").write_text(
    json.dumps(compiled, indent=2, allow_nan=False) + "\n"
)
rows = json.loads((out / "summary.json").read_text())
matched = []
arrays = {}
for index, row in enumerate(rows):
    if not row["paired_tomography"]:
        continue
    a = np.load(out / row["data"])
    shots = (64 * 4096) // len(a["states"])
    rng = np.random.default_rng(20261018 + 400 + index)
    counts = np.array([rng.multinomial(shots, p) for p in a["exact_cloud"]])
    arrays[str(index)] = counts
    matched.append(
        clean(
            {
                "data": row["data"],
                "shots_per_state": shots,
                "total_tomography_shots": int(shots * len(counts)),
                "total_triple_shots": 64 * 4096,
                "metrics": tomography_cloud_metrics(
                    counts / shots, a["ideal_cloud"], a["sampled_tensor"], 2
                ),
            }
        )
    )
np.savez_compressed(out / "matched_budget_counts.npz", **arrays)
(out / "matched_budget_summary.json").write_text(
    json.dumps(matched, indent=2, allow_nan=False) + "\n"
)
print(json.dumps(compiled, indent=2))
