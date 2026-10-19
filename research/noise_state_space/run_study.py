"""Reproduce the noise study; run from repository root with --quick for smoke."""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import platform
import sys
import time

import cirq
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cirq_sic.noise import (
    LocalMomentNoise,
    simulate_triple_tensor,
    simulate_sic_tomography,
    triple_tensor_diagnostics,
    compare_state_spaces,
    tomography_cloud_metrics,
)
from cirq_sic.measurements import sic_povm
from cirq_sic.triples import triple_products


def clean(value):
    if isinstance(value, dict):
        return {
            k: clean(v)
            for k, v in value.items()
            if k not in ("radii", "reference_radii", "directions")
        }
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(quick=False):
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    seed = 20261018
    rng = np.random.default_rng(seed)
    count = 48 if quick else 384
    states = rng.normal(size=(count, 2)) + 1j * rng.normal(size=(count, 2))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    reference = triple_products(sic_povm(2).effects * 2)
    cases = [("ideal", 0, "none", {})]
    channels = {
        "depolarizing": cirq.depolarize,
        "bit_flip": cirq.bit_flip,
        "phase_flip": cirq.phase_flip,
        "phase_damping": cirq.phase_damp,
        "amplitude_damping": cirq.amplitude_damp,
    }
    for name, factory in channels.items():
        for rate in ([0.05] if quick else [0.01, 0.05, 0.15]):
            channel = factory(rate)
            cases.append((name, rate, "preparation", {"preparation_channel": channel}))
            cases.append(
                (name, rate, "moment_all", {"noise": LocalMomentNoise(channel)})
            )
    for rate in ([0.05] if quick else [0.01, 0.05, 0.15, 0.2]):
        cases.append(("bit_flip", rate, "readout", {"readout_error": rate}))
    cases.extend(
        [
            ("coherent_z", 0.4, "preparation", {"preparation_channel": cirq.rz(0.4)}),
            (
                "bit_flip",
                0.05,
                "moment_control",
                {
                    "noise": LocalMomentNoise(
                        cirq.bit_flip(0.05), qubits=[cirq.LineQubit(0)]
                    )
                },
            ),
            (
                "phase_flip",
                0.05,
                "moment_control",
                {
                    "noise": LocalMomentNoise(
                        cirq.phase_flip(0.05), qubits=[cirq.LineQubit(0)]
                    )
                },
            ),
            (
                "bit_flip",
                0.05,
                "moment_register1",
                {
                    "noise": LocalMomentNoise(
                        cirq.bit_flip(0.05), qubits=[cirq.LineQubit(1)]
                    )
                },
            ),
        ]
    )
    rows = []
    start = time.time()
    for index, (name, rate, placement, kwargs) in enumerate(cases):
        result = simulate_triple_tensor(2, shots=4096, seed=seed + index, **kwargs)
        exact = replace(result, estimates=result.exact_expectations)
        # Targeted ancilla/register cases have no corresponding role in SIC tomography.
        paired = placement not in ("moment_control", "moment_register1")
        row = {
            "name": name,
            "rate": rate,
            "placement": placement,
            "paired_tomography": paired,
            "diagnostics": triple_tensor_diagnostics(
                exact.raw_tensor, 2, reference=reference
            ),
            "sampled_diagnostics": triple_tensor_diagnostics(
                result.raw_tensor, 2, reference=reference
            ),
            "geometry": compare_state_spaces(
                exact.tensor, reference, 2, directions=4096, seed=seed
            ),
            "sampled_geometry": compare_state_spaces(
                result.tensor, reference, 2, directions=4096, seed=seed
            ),
        }
        arrays = {
            "triplets": result.triplets,
            "exact_expectations": result.exact_expectations,
            "estimates": result.estimates,
            "counts_zero": result.counts_zero,
            "states": states,
            "tensor": exact.tensor,
            "sampled_tensor": result.tensor,
        }
        if paired:
            cloud = simulate_sic_tomography(
                states, shots=4096, seed=seed + 100 + index, **kwargs
            )
            row["tomography"] = tomography_cloud_metrics(
                cloud["exact"], cloud["ideal"], exact.tensor, 2
            )
            row["sampled_tomography"] = tomography_cloud_metrics(
                cloud["probabilities"], cloud["ideal"], result.tensor, 2
            )
            arrays.update(
                ideal_cloud=cloud["ideal"],
                exact_cloud=cloud["exact"],
                sampled_cloud=cloud["probabilities"],
                tomography_counts=cloud["counts"],
            )
        # Explicitly retain failed bootstrap geometries rather than dropping them.
        if placement in ("none", "readout") or (
            name == "depolarizing" and rate == 0.05
        ):
            volumes = []
            failed = 0
            for t in result.bootstrap(20 if quick else 100, seed=seed + 200 + index):
                metrics = compare_state_spaces(
                    t, reference, 2, directions=512, seed=seed
                )
                if metrics["status"] == "ok":
                    volumes.append(metrics["relative_volume"])
                else:
                    failed += 1
            row["bootstrap"] = {
                "replicates": 20 if quick else 100,
                "invalid_center": failed,
                "volume_percentiles_conditional_on_valid_center": (
                    np.percentile(volumes, [2.5, 50, 97.5]) if volumes else None
                ),
            }
        filename = f"{index:02d}_{name}_{placement}_{rate:g}.npz"
        np.savez_compressed(out / filename, **arrays)
        row["data"] = filename
        rows.append(clean(row))
        (out / "summary.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n"
        )
        print(
            index + 1,
            len(cases),
            name,
            rate,
            placement,
            row["geometry"]["status"],
            flush=True,
        )
    # Higher-dimensional smoke/cross-check with symmetry imposed to control cost.
    if not quick:
        d = 4
        ref = triple_products(sic_povm(d).effects * d)
        higher = []
        for name, kwargs in [
            ("ideal", {}),
            ("readout", {"readout_error": 0.01}),
            (
                "preparation_depolarizing",
                {"preparation_channel": cirq.depolarize(0.01)},
            ),
        ]:
            r = simulate_triple_tensor(d, ordered=False, **kwargs)
            metrics = compare_state_spaces(r.tensor, ref, d, directions=2048, seed=seed)
            higher.append(
                clean(
                    {
                        "name": name,
                        "diagnostics": triple_tensor_diagnostics(
                            r.tensor, d, reference=ref
                        ),
                        "geometry": metrics,
                    }
                )
            )
            np.savez_compressed(
                out / f"d4_{name}.npz",
                triplets=r.triplets,
                estimates=r.estimates,
                tensor=r.tensor,
            )
            print("d4", name, metrics["status"], flush=True)
        (out / "d4_summary.json").write_text(
            json.dumps(higher, indent=2, allow_nan=False) + "\n"
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "states": count,
                "shots_per_setting": 4096,
                "ordered_triples_d2": True,
                "d2_settings": 64,
                "d4_ordered": False,
                "d4_settings": 816,
                "quick": quick,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "cirq": cirq.__version__,
                "seconds": time.time() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
