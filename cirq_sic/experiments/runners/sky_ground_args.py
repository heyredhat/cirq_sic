"""Generate an explicit batch argument file without import-time side effects."""

import argparse
from pathlib import Path


def write_arguments(path, dataset_id, n_shots=100000):
    """Write the historical power-of-two hardware sweep and return its path.

    The file contains one command-line argument set per line. It does not submit
    jobs. Both hardware noise data and optional compilation tools are needed
    to execute the full sweep.
    """
    if n_shots < 1:
        raise ValueError("n_shots must be positive.")
    lines = []
    for d in (2, 4, 8):
        for run_type in ("clean", "noisy"):
            for optimizer in ("cirq", "bqskit_server_4"):
                for implementation in ("ak", "simple", "msimple"):
                    lines.append(
                        f"-dataset_id {dataset_id} -processor_id willow_pink "
                        f"-run_type {run_type} -n_shots {n_shots} -optimizer {optimizer} "
                        f"-d {d} -fiducial_description numerical_sic -wh_implementation {implementation}"
                    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    """Write argument sets to the path supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--n-shots", type=int, default=100000)
    args = parser.parse_args(argv)
    return write_arguments(args.path, args.dataset_id, args.n_shots)


if __name__ == "__main__":
    main()
