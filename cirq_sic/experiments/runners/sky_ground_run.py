"""Command-line runner for the legacy sky/ground task battery.

Run ``python -m cirq_sic.experiments.runners.sky_ground_run --help`` for options.
For arbitrary logical dimensions, use :mod:`cirq_sic.workflow`.
"""

import argparse
from pathlib import Path

import cirq
import numpy as np

from cirq_sic.circuits import d4_sic_fiducial
from cirq_sic.experiments.sky_ground import get_wh_qubits, run_sky_ground_tasks
from cirq_sic.sics import load_sic_fiducial
from cirq_sic.utils import rand_ket


def main(argv=None):
    """Parse command-line options and run a reproducible task battery."""
    parser = argparse.ArgumentParser(prog="sky_ground")
    parser.add_argument("-dataset_id", "--dataset-id", required=True)
    parser.add_argument("-processor_id", "--processor-id", default="willow_pink")
    parser.add_argument(
        "-run_type", "--run-type", choices=["clean", "noisy", "real"], default="clean"
    )
    parser.add_argument("-n_shots", "--n-shots", type=int, default=10000)
    parser.add_argument("-optimizer", "--optimizer", default="none")
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument(
        "-fiducial_description",
        "--fiducial-description",
        choices=["numerical_sic", "rand_ket", "d4_monomial"],
        default="numerical_sic",
    )
    parser.add_argument(
        "-wh_implementation",
        "--wh-implementation",
        choices=["ak", "simple", "msimple"],
        default="simple",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("data/sky_ground"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plots", action="store_true", help="Save Matplotlib comparison figures."
    )
    args = vars(parser.parse_args(argv))
    base_dir, seed, plots = (args.pop(name) for name in ("base_dir", "seed", "plots"))
    specs = args
    specs["qubits"] = get_wh_qubits(specs["d"], specs["wh_implementation"])
    if specs["fiducial_description"] == "numerical_sic":
        specs["fiducial"] = load_sic_fiducial(specs["d"])
    elif specs["fiducial_description"] == "rand_ket":
        specs["fiducial"] = rand_ket(specs["d"], rng=np.random.default_rng(seed))
    else:
        if specs["d"] != 4:
            parser.error("d4_monomial requires d=4.")
        specs["fiducial_circuit"] = d4_sic_fiducial
    sampler = (
        cirq.Simulator(seed=seed, dtype=np.complex128)
        if specs["run_type"] == "clean"
        else None
    )
    results = run_sky_ground_tasks(specs, sampler=sampler, base_dir=base_dir)
    if plots:
        from cirq_sic.experiments.sky_ground_analysis import sky_ground_images

        image_directory = base_dir / specs["dataset_id"] / f"d{specs['d']}" / "img"
        sky_ground_images(specs, base_dir=base_dir, img_dir=image_directory)
    return results


if __name__ == "__main__":
    main()
