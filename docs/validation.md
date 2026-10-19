# Local validation of the 0.2 refactor

Validated on October 13, 2026. All changes are local; no commits, pushes or live
hardware submissions were made.

## Reproduce

From a fresh environment at the repository root:

```bash
python -m pip install -e '.[test,notebooks]'
python -m pytest --cov=cirq_sic --cov-report=term-missing
python tools/execute_notebooks.py
python -m build
```

The prepared isolated environment used for this work is
`/tmp/cirq-sic-venv`. The existing environment at
`/home/heyredhat/anaconda3/envs/cirq_sic` was also used for compatibility checks.
These paths are local conveniences, not requirements or embedded library paths.

## Test results

| Environment | Result | Optional skip |
| --- | --- | --- |
| Python 3.9.13, Cirq 1.3.0 | 259 passed, 1 skipped | Cirq Google device-routing test; backend absent |
| Python 3.13, Cirq 1.6.1 | 259 passed, 1 skipped | Plotly visualization test; backend absent |

Each skipped test passed in the other environment. The measured statement
coverage in the isolated environment is **88%**, including legacy modules.
This is not a claim that every backend or hardware behavior has been tested.
Third-party deprecation warnings are recorded separately from test failures.

The suite checks:

- Complete Born distributions, isometry/unitary identities, column placement,
  custom qubit order, padding, and pure/mixed preparation; dimensions 1–8 are
  represented, including nonpowers of two.
- Legacy embedded modular gates and Arthurs–Kelly distributions on both logical
  and padded inputs; Gray-code preparation and conjugation.
- SIC refinement and reconstruction, general IC inversion, non-IC rejection,
  physical projection, and complex mixed states.
- Tensor-Pauli versus cyclic-WH conventions, known stabilizer/T-state values,
  direct mixed WH overlap distributions, purity and entropy identities.
- Calibrated sky/ground residuals, singular calibration, uncertainty propagation,
  noise timing, idle qubits, explicit noisy sampling and layout validation.
- Real/imaginary triple signs, mixed-state cycle tests, tensor symmetries,
  moments, positivity reconstruction and 2D/3D section geometry.
- Seeded sampling, complete integer counts, artifact round trips, content
  fingerprints, wrong-plan rejection, and metadata/schema validation.
- Full-unitary synthesis for random, diagonal and identity matrices, and the
  compiled qutrit SIC's probabilities.
- Legacy task persistence, runners, optimizers' readout mappings, plotting,
  import side effects, and the public API.

## Demonstrations and documentation

All six maintained notebooks execute from fresh kernels using the isolated
interpreter. Saved outputs were refreshed. The tutorial has four embedded
figures, visually checked for complete labels and layout. The embedding
notebook checks dimensions 2–8; other notebooks cover mixed entropy, fixed-SIC
outcome entropy, experiment artifacts, sky/ground data and triple geometry.

All 13 Python examples in the API guide executed successfully together.
Documentation links were checked. Mathematical documentation includes
conventions, derivations, primary references and estimator limitations.

## Distribution and historical-data checks

Both wheel and source distribution build successfully. The installed wheel was
imported from `/tmp`, outside the checkout, and passed qutrit tomography, direct
mixed WH entropy, persistence/fingerprint and largest-fiducial resource checks.

The wheel contains all **149** numerical fiducials (dimensions 2–151 except 72)
and excludes notebooks, research data, tests and bytecode. The source
distribution includes the six maintained notebooks and their execution tool.
All fiducial files have the expected finite, nonzero `(d,2)` data layout.

All **64** historical task JSON records examined load through the compatibility
codec without ReCirq. All **2,122** files moved from the original notebook tree
to `research/notebooks` match their original Git blobs byte for byte. Original
research outputs were preserved rather than re-executed or overwritten.

## Limits of this verification

No real quantum device was contacted for execution. Google routing was tested
with local device metadata; BQSKit readout permutations were tested with a
compiler stub, not an actual synthesis service. Large-dimension gate synthesis
and noisy density simulation were not benchmarked. Dense scaling is explicitly
documented and can become prohibitive well before the largest bundled dimension.

The research archive is preserved historical material. Its exploratory
notebooks, external-data assumptions, animation workloads, and old APIs are
not part of the maintained fresh-kernel suite. CI configuration was added
locally; no remote CI run has occurred because nothing has been pushed.

## Noise reconstruction pipeline

`tests/test_noise.py` validates the noise API against independent Kraus maps,
analytic qubit radial/volume formulas, amplitude-damping affine parameters and
common-unitary invariance. It also checks compiled circuits, raw register
asymmetry, readout confusion, seeded finite shots, setting-aware bootstrap,
center failure, singular clouds and input/resource guards. All 283 repository
tests passed after this addition. The accompanying study records exact-channel,
finite-shot and Monte Carlo uncertainty separately; it is not hardware validation.
