# API and experiment guide

`cirq_sic` separates measurement definitions, circuit construction, execution,
and numerical analysis. Start with `workflow` for complete experiments, then
use the lower-level modules when you need custom preparations or hardware
integration. All examples use NumPy arrays and ordinary Cirq circuits.

## Modules

| Module | Purpose |
| --- | --- |
| [`measurements`](../cirq_sic/measurements.py) | Rank-one POVMs, exact qubit embeddings, preparation, and direct WH overlap circuits. |
| [`workflow`](../cirq_sic/workflow.py) | The four experiment families, exact probabilities, sampling, compilation, and JSON persistence. |
| [`analysis`](../cirq_sic/analysis.py) | Tomography, characteristic functions, entropy diagnostics, and Born-rule consistency. |
| [`triples`](../cirq_sic/triples.py) | Complex triple products, three-copy cycle tests, and probability state-space reconstruction. |
| [`geometry`](../cirq_sic/geometry.py) | Two- and three-dimensional sections of triple-derived positivity constraints; plotting imports are optional. |
| [`wh`](../cirq_sic/wh.py) | Cyclic Weyl–Heisenberg operators, orbit frames, and effect arrays. |
| [`sics`](../cirq_sic/sics.py) | Bundled fiducials, SIC refinement/search, and frame-potential diagnostics. |
| [`circuits`](../cirq_sic/circuits.py) | Composable Fourier/displacement gates and compatibility WH/Arthurs–Kelly circuits. |
| [`ansatz`](../cirq_sic/ansatz.py) | Gray-code state-preparation angles and unitary reconstruction. |
| [`utils`](../cirq_sic/utils.py) | Random states, partial traces, register probability helpers, and circuit SVG export. |
| [`experiments`](../cirq_sic/experiments/__init__.py) | Compatibility task runners, stored task processing, and older sky/ground analyses. New experiments should use `workflow`. |

Module and function docstrings give argument shapes, conventions, and error
conditions. For example, `help(cirq_sic.Measurement)` and
`help(cirq_sic.triple_product_state_space)` work after `import cirq_sic`.
Top-level public names are loaded lazily; prefer explicit imports in library
code. Third-party names such as `np` and `cirq` are not re-exported.

## States, effects, and labels

A logical state is a normalized complex ket of shape `(d,)` or a Hermitian,
positive semidefinite, trace-one density matrix of shape `(d, d)`. Preparation
and measurement APIs reject malformed or unnormalized inputs. Normalize
intentionally when constructing your state:

```python
import cirq
import numpy as np

psi = np.array([1, 1j, 2], dtype=complex)
psi /= np.linalg.norm(psi)
rho = 0.7 * np.outer(psi, psi.conj()) + 0.3 * np.eye(3) / 3
```

All registers use **big-endian** qubit order. A logical dimension `d` occupies
basis states `0` through `d - 1` of its system register. The measurement
circuit appends ancillas initially in zero. For WH measurements,

```text
X |j> = |j + 1 mod d>
Z |j> = exp(2*pi*i*j/d) |j>
D[a,b] = X**a @ Z**b
outcome k = a*d + b
```

A cyclic dimension-four WH basis differs from the tensor-product two-qubit
Pauli basis. Encoding a qudit in qubits does not change this mathematical
convention.

## Define a measurement

```python
from cirq_sic import Measurement, sic_povm, wh_measurement

sic = sic_povm(3)
wh = wh_measurement(psi)
basis = Measurement(np.eye(3), name="computational basis")
p = sic.probabilities(rho)
```

`Measurement(frame, name="POVM", *, atol=1e-9)` takes a **weighted** frame
of shape `(d, m)`. Column `f_k` defines `E_k = np.outer(f_k, f_k.conj())`,
and completeness requires `frame @ frame.conj().T = I`.
Do not pass unscaled unit-norm WH orbit vectors: `wh_measurement` performs
the required division by `sqrt(d)`.

| Property or method | Meaning |
| --- | --- |
| `.dimension`, `.num_outcomes` | Logical input dimension `d` and number of effects `m`. |
| `.frame` | Read-only weighted columns, shape `(d, m)`. |
| `.effects` | Read-only effect matrices, shape `(m, d, d)`. |
| `.states` | Read-only unit-norm columns, shape `(d, m)`. Iterate over `.states.T` for reference preparations. |
| `.num_system_qubits` | `ceil(log2(d))`, with a minimum of one qubit. |
| `.num_qubits` | Total measured qubits, `ceil(log2(m))`, with a minimum of one. |
| `.num_ancilla_qubits` | Difference between total and system qubit counts. |
| `.probabilities(state)` | Logical Born probabilities, shape `(m,)`, for a ket or density matrix. |
| `.isometry()` | Padded measurement map, shape `(2**num_qubits, d)`. |
| `.unitary()` | Read-only full dilation in system-then-ancilla input order. |
| `.is_sic(atol=1e-7)` | Check equal weights and SIC overlaps; this generic check allocates the complete overlap matrix. |
| `.circuit(state=None, **kwargs)` | Convenience call to `build_measurement_circuit`. |

`wh_measurement` accepts any unit fiducial, but the resulting POVM need not
be informationally complete. Tomography checks this property separately.
`sic_povm(d=None, *, fiducial=None, atol=1e-7)` validates SIC overlaps and
accepts either a bundled dimension or your supplied fiducial. SIC existence
in every dimension is not assumed: the embedding works for any supplied
valid measurement.

Bundled files cover dimensions 2–151 except 72. `sic_povm(1)` supplies the
trivial one-dimensional SIC. Some historical files have modest precision;
`sic_povm` refines and caches bundled dimensions through 32. The raw
`load_sic_fiducial(d)` loader only normalizes the original data. Use
`refine_sic_fiducial(fiducial, tolerance=...)` explicitly for other approximate
fiducials, or choose an explicit approximation tolerance when appropriate.

### Probability measurement versus state update

A POVM specifies outcome probabilities; it does not uniquely specify the
post-measurement quantum state. The default Naimark implementation measures
its complete output register and does not expose a surviving logical system.
It is suitable for destructive tomography and probability experiments.

In sky/ground calibration, `.states.T` provides explicitly prepared reference
states. This is a preparation choice in that protocol, not a promise that an
arbitrary measurement apparatus implements the same instrument. The legacy
Arthurs–Kelly circuits have a separate remaining system and their own state
update; do not infer that update from the POVM effects alone.

## Build and inspect circuits

```python
from cirq_sic import build_measurement_circuit

built = build_measurement_circuit(sic, rho, key="sic")
print(built.circuit)
print(built.qubit_order)
```

The returned `MeasurementCircuit` contains:

- `.circuit` and `.measurement`;
- `.system_qubits`, `.ancilla_qubits`, and `.environment_qubits`;
- `.measured_qubits`, in the exact order written to `.measurement_key`;
- `.qubit_order`, containing measured qubits followed by the environment;
- `.outcome_indices`, the contiguous valid integers `0` through `m - 1`.

Pure preparation uses a unitary on the system. Mixed preparation constructs
`sum_i sqrt(lambda_i) |v_i>_system |i>_environment` and leaves the environment
unmeasured. It requires one environment qubit per system qubit. Measurement
ancillas and environment qubits are distinct registers.

`prepare_state(state, system_qubits, environment_qubits=...)` returns a
preparation operation for initially zero registers. `embed_state(state,
num_qubits=None)` instead returns a padded ket or density matrix, useful for
Cirq's simulator `initial_state` argument. `ansatz_circuit(ket)` offers a
Gray-code preparation generator using one-qubit rotations and CNOTs, with
`conjugate=True` support.

### Existing or unknown physical inputs and custom qubits

Omit `state` when the system already contains the physical state to measure.
The library then adds no preparation. Specify the system and measurement
ancillas explicitly:

```python
system = tuple(cirq.NamedQubit(f"s{i}") for i in range(2))
ancillas = tuple(cirq.NamedQubit(f"a{i}") for i in range(2))
readout = build_measurement_circuit(
    sic, system_qubits=system, ancilla_qubits=ancillas, key="sic"
)
```

Append `readout.circuit` after your preparation/device process. The system
must be supported on its logical dimension-three subspace and the ancillas
must start at zero. Use `readout.qubit_order` explicitly in simulations;
alphabetical qubit sorting would reorder these registers.

To use workflow persistence and analysis with this custom circuit, wrap it:

```python
from cirq_sic import CircuitSpec, Experiment

spec = CircuitSpec(
    name="measurement", circuit=readout.circuit,
    qubit_order=readout.qubit_order,
    measured_qubits=readout.measured_qubits,
    key=readout.measurement_key, num_outcomes=sic.num_outcomes,
)
custom = Experiment(
    [spec], name="device tomography",
    metadata={"kind": "tomography", "dimension": sic.dimension},
    arrays={"effects": sic.effects},
)
```

Include your preparation circuit in `spec.circuit` when it is part of the
executable program. Simulating the readout alone starts the system in zero;
it cannot supply an unknown laboratory state to a simulator.

## The four workflows

### SIC tomography

```python
from cirq_sic import tomography_experiment

experiment = tomography_experiment(rho, measurement=sic)
exact = experiment.simulate()
report = experiment.analyze(exact)
assert np.allclose(report["density_matrix"], rho)
```

`report` includes the reconstructed density matrix, measured probabilities,
purity, minimum eigenvalue, and whether physical projection was requested.
`linear_inversion(probabilities, effects)` also works directly with an
informationally complete effect array. `sic_reconstruction` implements the
closed SIC inverse from unit-trace projectors.

Finite-shot linear estimates can have negative eigenvalues. Passing
`physical=True` projects onto positive trace-one matrices; that changes the
estimator and is not maximum-likelihood reconstruction. Keep raw frequencies
and unprojected diagnostics when assessing reconstruction error.

### WH entropy, including mixed states

```python
from cirq_sic import wh_entropy_experiment, entropy_experiment

entropy_job = wh_entropy_experiment(rho)
entropy_report = entropy_job.analyze(entropy_job.simulate())
print(entropy_report["entropy"], entropy_report["purity"])
```

The direct method prepares `rho` and its computational-basis conjugate, then
measures a generalized Bell basis. Its outcome distribution is
`s_ab = Tr(rho D_ab rho D_ab†)/d`; purity is `d*s_00`, and the purity-corrected
Rényi-2 diagnostic is `-log_base(sum(s**2)/s_00)`. This needs a description
of the conjugate preparation, not a universal conjugation operation on an
unknown state. `bell_overlap_circuit` exposes the underlying circuit, while
`wh_entropy_from_overlaps` analyzes its distribution directly.

The Bell circuit uses two separately embedded system registers. Its
`.measurement` describes a Bell basis in abstract dimension `d²`, while its
actual measured register can use one extra qubit compared with a compact
dimension-`d²` encoding. Always use the returned `.measured_qubits` and
`.qubit_order` for this specialized circuit.

For other Rényi orders, or the tensor-Pauli convention, use tomography:

```python
inferred = entropy_experiment(rho, measurement=sic, basis="wh", alpha=3, mixed=True)
inferred_report = inferred.analyze(inferred.simulate())
```

`basis="pauli"` requires a power-of-two logical dimension. `basis="wh"`
uses the cyclic logical dimension. The documented mixed extension is a
purity-corrected diagnostic; it is not a faithful magic monotone for all
mixed states. These nonlinear plug-in estimates have finite-shot bias.

### Sky/ground Born-rule consistency

```python
from cirq_sic import sky_ground_experiment

battery = sky_ground_experiment([psi, rho], reference=sic, ground=basis)
diagnostics = battery.analyze(battery.simulate())
assert np.max(np.abs(diagnostics["residual"])) < 1e-9
```

The battery independently measures input sky and ground distributions and
calibrates both devices on each normalized reference state. Analysis compares
`q` with `C @ inv(P) @ p`, where columns of `C` and `P` correspond to prepared
reference states. For an ideal SIC, `inv(P) = (d+1)*I - J/d`.

The reference must be a minimal informationally complete rank-one POVM.
Diagnostics preserve negative predictions and include residuals and error
norms; no clipping makes inconsistent hardware data appear consistent.
`born_rule_diagnostics` additionally accepts shot counts for an independent
multinomial uncertainty estimate. That estimate does not include drift or
uncertainty in a fitted reference inverse.

### Triple products and probability state space

```python
from cirq_sic import (
    triple_product_experiment, triple_product_state_space,
    sic_state_constraints,
)

qubit_sic = sic_povm(2)
triples_job = triple_product_experiment(qubit_sic.states.T)
triples_report = triples_job.analyze(triples_job.simulate())
tensor = triples_report["tensor"]
pencil = triple_product_state_space(tensor, dimension=2)
probabilities = qubit_sic.probabilities(np.array([1, 0]))
constraints = sic_state_constraints(probabilities, tensor, dimension=2)
assert constraints["is_physical"]
```

Each cycle test uses three preparations and a control qubit; separate control
measurements recover real and imaginary components of `Tr(rho_i rho_j rho_k)`.
The default measures all unordered triples, then expands cyclic/conjugate
symmetries. With `m` states this requires `2*binomial(m+2,3)` circuits. Supply
`triplets=[(i,j,k), ...]` to inspect a subset; partial experiments return
`estimates` without claiming a complete tensor.

For exact SIC triple data, `sum_i p_i*pencil[i]` is positive semidefinite
exactly when normalized probabilities `p` describe a quantum state.
`geometry` can plot sections of this positivity region. Noisy triple data
produce an estimated constraint region, not a certificate of a valid SIC.
State-space reconstruction does not determine a unique Hilbert-space
orientation or fiducial.

## Execute, compile, save, and iterate

Every workflow returns `Experiment` with the same methods:

```python
exact = experiment.simulate()
noisy = experiment.simulate(noise=cirq.depolarize(0.001))
sampled = experiment.run(repetitions=5000, seed=17)

sampler = cirq.Simulator(seed=17)  # Replace with your configured hardware sampler.
sampled_again = experiment.run(sampler, repetitions=5000)
```

A supplied sampler owns its own randomness; do not also pass `seed` to
`.run`. Circuits contain dense reference gates initially. Compile before
using a device that does not accept them:

```python
from cirq_sic import compile_circuit

compiled = experiment.transform(compile_circuit)
```

Compilation synthesizes unitaries and targets Cirq's CZ gate set by default;
`compile_circuit(circuit, gateset=...)` accepts another target. It does not
perform device placement or connectivity routing. `transform` must preserve
the qubits and terminal measurement layout. If routing changes them, create
new `CircuitSpec` objects with the updated explicit orders.

```python
from cirq_sic import ExperimentResult

experiment.save("artifacts/tomography.json")
sampled.save("artifacts/tomography-result.json")
restored = Experiment.load("artifacts/tomography.json")
restored_result = ExperimentResult.load("artifacts/tomography-result.json")
restored_report = restored.analyze(restored_result, physical=True)
```

Artifacts use versioned JSON, standard Cirq serialization, explicit layouts,
and numeric analysis inputs. No executable pickle or task object is needed.
You can save before execution, rerun a loaded experiment with another sampler,
and analyze a saved result without rerunning circuits.

`experiment.fingerprint` identifies the recorded circuits, layouts, metadata,
and analysis inputs. Generated results carry that fingerprint, and analysis
rejects results from a different experiment. In particular, analyze results
from `compiled.run(...)` with `compiled.analyze(...)`. For independently
verified imported results that have no fingerprint, explicitly pass
`allow_unbound=True`; an existing mismatched fingerprint is always rejected.
The fingerprint checks artifact consistency, not whether a hardware backend
actually executed the recorded circuit.

### Reproducible noise timing

`noisy = experiment.with_noise(cirq.depolarize(0.01))` creates a new plan
containing explicit channel operations on the original circuit timeline,
including recorded idle qubits. Use `noisy.simulate()` and
`noisy.run(repetitions=1000, seed=7)` to compare exact and sampled data from
the same noisy circuit. Analyze those results with `noisy`, since adding noise
changes the circuit fingerprint. Padding leakage is still retained.

`experiment.simulate(noise=...)` uses the same timing for a one-off exact
calculation. Noise inserted after the recorded readout cannot change its
already-recorded probabilities; noise before it and genuine idle moments are
preserved. A supplied sampler owns its own noise scheduling, which need not
match explicit circuit noise. In particular, do not assume a simulator's
implicit noise option adds channels at precisely the same boundaries. Noise
on abstract dense gates is also not a device-depth model: compile first when
you intend a per-elementary-gate or per-moment hardware approximation.

### Results and padded outcomes

`result["measurement"]` is an `OutcomeData` object for single-measurement
workflows; multi-circuit results expose all names through `result.outcomes`.

| Attribute | Interpretation |
| --- | --- |
| `.probabilities` | Full binary output distribution, including padding; sums to one. |
| `.valid` | First `num_outcomes` entries, with no conditional renormalization. |
| `.leakage` | Total mass in padded outputs. |
| `.counts`, `.repetitions` | Integer samples and shot count, or `None` for exact simulation. |
| `.standard_errors` | Marginal plug-in binomial errors for sampled probabilities, or `None`. |

Built-in analysis rejects padding leakage above `leakage_tolerance` rather
than postselecting it away. Inspect full counts and leakage when diagnosing
hardware. Standard errors are not confidence intervals; they can be zero at
an observed zero count despite nonzero uncertainty.

## Resource limits and compatibility

The dense Naimark unitary has dimension at most twice the number of outcomes,
so storage is O(m²) and generic completion/synthesis can be expensive. For a
SIC, `m=d²`; the dense matrix therefore uses O(d⁴) memory. Purifications,
Bell overlaps, and triple tests require additional qubits. Exact density
simulation with noise costs much more memory than pure state-vector
simulation. Mathematical support for arbitrary dimension does not imply
practical compilation or execution at every dimension.

Legacy `simple_wh_povm_d` uses separate embedded registers, with outcome
integer `a*2**n+b` and interleaved padding. Its `aux` argument is retained
but no longer needed. The unified measurement API uses contiguous `a*d+b`
outcomes instead. Keep the circuit's recorded layout with its data; do not
mix these two decoding conventions.
