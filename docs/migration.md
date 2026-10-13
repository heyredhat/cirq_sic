# Migrating from 0.1

Version 0.2 adds a common experiment workflow while retaining the original
circuit generators and task classes where their contracts are meaningful.
Nothing is uploaded, submitted to hardware, committed, or pushed by the new
workflow automatically.

## Prefer experiment plans for new work

| Previous entry point | New entry point |
| --- | --- |
| Notebook-specific embedding task | `sic_povm(d)` + `tomography_experiment(state, measurement)` |
| `WHPOVMOnStatesTask` | `tomography_experiment` or a custom `CircuitSpec` |
| Hand-written entropy formulas | `wh_entropy_experiment` for direct cyclic WH Rényi-2, or `entropy_experiment` |
| ReCirq sky/ground battery | `sky_ground_experiment(states, reference, ground)` |
| Triple-product notebook circuit/cache | `triple_product_experiment(states, triplets=...)` |
| Ad hoc JSON result directories | `Experiment.save/load`, `ExperimentResult.save/load` |
| Wildcard-provided `np`, `cirq`, `sc`, `mpow` | Explicit imports of NumPy, Cirq, SciPy and `numpy.linalg.matrix_power` |

Public library names remain available at the package root through a lazy export
map. Third-party names are no longer an accidental part of the public API.
Imports no longer configure NumPy or initialize optional compiler/GPU stacks.
Use `python tools/update_exports.py` after deliberately changing public exports.

## Corrected arbitrary-dimensional behavior

The former `Z_d`/`X_d` implementation relied on auxiliary-register arithmetic
that did not implement the requested embedded operators reliably. Embedded
single-register gates now act as the logical unitary on levels `< d` and the
identity on unused levels. Controlled modular gates act only when both logical
register values are valid. Compatibility auxiliary arguments remain accepted;
these corrected dense implementations do not need scratch qubits.

For new measurements use `build_measurement_circuit`. Its outcome label is
always contiguous `a*d+b`. Legacy `simple_wh_povm_d` measures two separate
binary registers and therefore retains the register label `a*2**n+b`.
Do not mix these raw indices. General probability helpers keep missing/padded
mass explicit rather than silently renormalizing.

The legacy task classes remain restricted to powers of two and now reject
other dimensions at construction. This avoids their old implicit floor-log2
behavior. The shared workflow supports arbitrary logical dimensions.

## Numerical fixes

- `load_sic_fiducial` returns normalized original data. `sic_povm` validates SIC
  overlaps and locally refines bundled dimensions through 32; this corrects
  finite-precision data that previously invalidated strict SIC formulas.
- `all_d2_fiducials()` now returns the intended two antipodal tetrahedra.
- Frame-potential weights use squared column norms, consistent with frame
  probabilities. `wh_2_frame_potential` retains its historical squared-excess
  objective; it is not the raw frame potential.
- State preparation validates normalization. Gray-code angle helpers require
  power-of-two lengths; `ansatz_circuit` pads arbitrary logical kets.
  `prepare_state` also supports mixed density matrices through purification.
- Geometry no longer mistakes the number of probabilities for the number of
  section directions; complex Hermitian pencils are supported.
- Rényi entropy bases/conventions and mixed-state assumptions are explicit.
  A finite-shot estimate need not be a pure state. Use `mixed=True` for the
  tomographic mixed extension and choose projection intentionally.

## Legacy task persistence and optimizers

The compatibility task loader reads historical `recirq.sky_ground` JSON through
an internal Cirq codec; ReCirq is not a core dependency. Newly run tasks preserve
raw measurements. Failed/missing data are not broadly swallowed. Reusing a
filename for incompatible task specifications raises instead of silently
reusing a stale result. Original archived datasets have not been rewritten.

For local compatibility tasks, `optimizer="none"` preserves the circuit and
`optimizer="cirq_generic"` uses a device-independent Cirq pass. A device-specific
optimizer requires its optional backend. Hardware samplers are supplied
explicitly; the library does not select a real processor behind your back.
Dynamic feed-forward circuits retain their separate exact reference circuit
where needed. The new shared workflow accepts static circuits with a single
terminal measurement; legacy semiclassical routines remain in the circuit API.

## Notebooks and dependencies

Maintained demonstrations now live in `notebooks/`, outside the installable
package. `Tutorial.ipynb` is the starting point; the embedding, mixed entropy,
SIC entropy, task and triple notebooks are focused follow-ups. Original
exploratory notebooks, animations, caches and outputs were moved intact into
`research/notebooks/`. Their historical dependencies and assumptions are not
part of the supported examples.

Install `.[notebooks,test]` for tutorial development, or only `.` for the numeric
and circuit API. JAX, Qiskit, CVXPY, Plotly, BQSKit, ReCirq, qsim and Cirq Google
are not required to import or use the core workflows. Plotly/BQSKit/device
adapters import their backend only when invoked.
