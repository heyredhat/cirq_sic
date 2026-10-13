"""Execute maintained notebooks with this interpreter, using isolated kernels.

Run from any directory. By default executed copies go to artifacts/notebooks;
--write refreshes source notebooks. A failure is never written over its source.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def execute(path):
    """Return a notebook executed with a kernel matching the current Python."""
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}
    )
    try:
        return client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()


def main():
    """Execute selected paths, or every maintained root notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--write", action="store_true", help="Refresh source notebook outputs"
    )
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cirq-sic-matplotlib")
    )
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    paths = args.paths or sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        executed = execute(path)
        output = path if args.write else ROOT / "artifacts" / "notebooks" / path.name
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, output)
        print(f"  Passed ({len(executed.cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
