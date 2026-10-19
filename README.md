# cirq_sic

SIC and Weyl–Heisenberg measurements on qubit hardware, with a shared workflow
for constructing circuits, simulating or sampling them, saving experiments,
and analyzing probability data.

The library supports four connected experiments:

| Experiment | Builder | Analysis |
| --- | --- | --- |
| SIC/IC state tomography | `tomography_experiment` | Linear inversion, optional physical projection |
| Stabilizer Rényi entropy | `wh_entropy_experiment`, `entropy_experiment` | Direct WH two-copy Rényi-2 estimate or tomography-based Pauli/WH entropy; mixed inputs supported |
| QBist sky/ground consistency | `sky_ground_experiment` | Calibrated Born-rule predictions and residuals across input states |
| Triple-product characterization | `triple_product_experiment` | Complex triple tensor, SIC moments, reconstructed probability state-space positivity |

Start with the executed **[tutorial notebook](notebooks/Tutorial.ipynb)**. It
walks through all four use cases, shows probability and state-space plots, saves
and reloads experiments, and compiles a qutrit SIC into one-/two-qubit gates.

## Install and test locally

Python 3.9 or later is supported. Use a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebooks]'
python -m pytest
python tools/execute_notebooks.py
jupyter lab notebooks/Tutorial.ipynb
```

The notebook runner executes each notebook in a fresh kernel using the Python
interpreter that invokes it. Executed copies go into `artifacts/notebooks`;
add `--write` to refresh the source notebooks after editing them. Tests and
examples use seeded randomness and temporary data directories.

A minimal runtime install is `python -m pip install -e .`: NumPy, SciPy, and
`cirq-core`. Optional extras are `notebooks`, `test`, `optimize` (BQSKit), and
`hardware` (Cirq Google/qsim). Importing `cirq_sic` does not load these optional
integrations, initialize JAX, change NumPy printing, or connect to hardware.

## A qutrit SIC in a few lines

```python
import numpy as np
from cirq_sic import Experiment, sic_povm, tomography_experiment

psi = np.array([1, 1j, -0.5], dtype=complex)
psi /= np.linalg.norm(psi)
plan = tomography_experiment(psi, sic_povm(3))

exact = plan.simulate()
observed = plan.run(repetitions=10_000, seed=7)
rho = plan.analyze(observed, physical=True)["density_matrix"]

plan.save("artifacts/qutrit-plan.json")
observed.save("artifacts/qutrit-data.json")
restored = Experiment.load("artifacts/qutrit-plan.json")
```

All experiment builders return an `Experiment`. Its named `CircuitSpec` objects
contain ordinary Cirq circuits and explicit register/measurement layouts.
`simulate()` returns exact distributions; `run()` accepts any `cirq.Sampler`.
`plan.with_noise(model)` creates an explicit noisy circuit plan shared by exact
and sampled execution. Saved plans and results use versioned JSON, preserving numeric analysis inputs,
raw counts, output order, and execution provenance.

## Any logical dimension, explicit encoding

The new measurement construction embeds the first `d` basis states into
`ceil(log2(d))` system qubits. For a rank-one POVM with `m` outcomes it constructs
a Naimark unitary on `ceil(log2(m))` measured qubits. WH outcome `k = a*d + b`
always means `D[a,b] = X**a @ Z**b`. A qutrit SIC therefore has outcomes 0–8 in
a 16-output binary register. Outcomes 9–15 are retained as leakage, never
postselected away or merged into valid outcomes.

A density matrix is prepared by a unitary purification with a separate
unmeasured environment. The same probability convention applies to pure and
mixed inputs. For an already prepared unknown input, use
`build_measurement_circuit(measurement, state=None, ...)`.

```python
from cirq_sic import compile_circuit

compiled = plan.transform(compile_circuit)
# Inspect gate counts and perform device placement/routing before hardware use.
# observed = compiled.run(your_sampler, repetitions=10_000)
```

This is an exact **dense reference implementation**, not a low-depth synthesis
claim: a SIC dilation needs O(d⁴) matrix storage and roughly O(d⁶) dense completion
work. Hardware feasibility depends on compiled depth and device connectivity.
The encoding has no power-of-two restriction. SIC existence in all dimensions
is not assumed: bundled fiducials cover dimensions 2–151 except 72, and supplied
valid fiducials/frames are supported. Some original files are approximate;
`sic_povm` refines and caches bundled dimensions through 32 before validation.
`load_sic_fiducial` preserves the original normalized data.

## Scientific conventions that matter

- Cyclic-qudit WH entropy and tensor-product Pauli entropy differ for `d >= 4`.
  Choose `basis="wh"` or `basis="pauli"` explicitly for tomographic entropy.
- The mixed-state purity-corrected entropy is a defined diagnostic, not a
  faithful magic measure on arbitrary mixtures of stabilizer states. Direct
  two-copy estimation requires preparation of the conjugate state.
- Finite-shot inversions, collision estimates, ratios, and logarithms have
  different statistical properties. The library keeps raw data and reports
  residuals without clipping negative predictions or discarding leakage.
- A POVM specifies outcome probabilities, not a unique post-measurement state.
  The default destructive Naimark measurement does not promise the
  Arthurs–Kelly instrument's state update.
- Triple-derived geometry from noisy measurements is an estimate. Complete
  positivity is checked through a matrix pencil; purity/cubic bounds alone
  are not a general mixed-state positivity test.

See [mathematical conventions](docs/mathematics.md) for formulas and references,
[the API guide](docs/api.md) for shapes and examples, and [migration notes](docs/migration.md)
for compatibility details. The [validation report](docs/validation.md) records
local test results, notebook execution, packaging checks and remaining limits.

## Repository layout

```text
cirq_sic/
    measurements.py    POVMs, embeddings, preparation, Bell overlaps
    workflow.py        experiment plans, execution, persistence, compilation
    analysis.py        tomography, entropy, Born-rule diagnostics
    triples.py         cycle tests and state-space reconstruction
    geometry.py        section geometry and optional plotting
    wh.py, sics.py      operators, fiducials, numerical diagnostics
    circuits.py        composable and compatibility circuit primitives
    ansatz.py, utils.py state preparation and numerical helpers
    experiments/       compatibility task runners and analyses
    sic_povms/         bundled numerical fiducials
notebooks/             maintained, executable demonstrations
tests/                 mathematical and workflow regressions
cirq_sic/tests/        retained compatibility regressions
docs/                  API, mathematics, migration and validation
research/notebooks/    original research notebooks and saved outputs
```

The original exploratory notebooks and their data are preserved in
[`research/notebooks`](research/README.md). They are historical records with
external-data and old-API assumptions; they are not the maintained demonstration
suite. The separate Bell-inequality studies and triple-product notes remain
research material.

### Noise and reconstructed state spaces

Use [`cirq_sic.noise`](docs/noise_state_space.md) to simulate noisy real SIC
triple circuits, compare their inferred geometry with an ideal SIC, and contrast
it with random-state SIC tomography. The module preserves raw ordered estimates,
separates channel bias from shot noise, and reports inconsistent or infeasible
reconstructions. A reproducible [research study and LaTeX report](research/noise_state_space/README.md)
cover depolarization, bit/phase flips, damping, readout and coherent errors.
