# Noise and reconstructed state spaces

The `cirq_sic.noise` module provides small-system circuit simulations and geometry
diagnostics. It compares two different experimental objects:

1. A positivity pencil formed from measured real triples and the **ideal SIC Gram
   matrix**. Noise can make this inconsistent with any SIC, or even infeasible.
2. A cloud of SIC outcome probabilities from known input states. This includes
   preparation and measurement errors and need not describe physical states under
   ideal linear inversion.

Neither is, by itself, an identifiable hardware noise model. Always retain the
raw measurements and describe the circuits, compiler, placement and shot budget.
The [research report](../research/noise_state_space/report/main.tex) gives analytic
counterexamples, reproducible results, limitations and a proposed hardware protocol.

## Example

```python
import cirq
import numpy as np
from cirq_sic import (
    sic_povm, triple_products, LocalMomentNoise, simulate_triple_tensor,
    simulate_sic_tomography, triple_tensor_diagnostics,
    compare_state_spaces, tomography_cloud_metrics,
)

measurement = sic_povm(2)
truth = triple_products(2 * measurement.effects)
experiment = simulate_triple_tensor(
    2, preparation_channel=cirq.depolarize(0.05),
    shots=4096, seed=42, ordered=True,
)
checks = triple_tensor_diagnostics(experiment.raw_tensor, 2, reference=truth)
geometry = compare_state_spaces(experiment.tensor, truth, 2, seed=42)
if geometry['status'] == 'ok':
    print(geometry['relative_volume'], geometry['jaccard'])
else:
    print('Uniform center is not strictly feasible; radial metrics unavailable.')

rng = np.random.default_rng(42)
states = rng.normal(size=(256, 2)) + 1j*rng.normal(size=(256, 2))
states /= np.linalg.norm(states, axis=1, keepdims=True)
cloud = simulate_sic_tomography(
    states, preparation_channel=cirq.depolarize(0.05), shots=4096, seed=43,
)
comparison = tomography_cloud_metrics(
    cloud['probabilities'], cloud['ideal'], experiment.tensor, 2,
)
replicates = experiment.bootstrap(100, seed=44)
```

`exact_expectations` separates channel bias from finite shots. `counts_zero`,
`triplets`, `shots` and `estimates` preserve the independently sampled settings.
The shot simulation draws binomials/multinomials from exact density-matrix circuit
probabilities, which has the same distribution as repeated terminal measurements
for these memoryless circuits. It does not simulate temporal drift or correlations
between shots. The result objects and arrays can be saved using NumPy/JSON.

## Noise placement and circuit conventions

- `preparation_channel`: any one-qubit channel or unitary, applied once to each
  system qubit **after** ideal preparation. No control or measurement-ancilla noise.
- `noise`: a Cirq noise model. `LocalMomentNoise(cirq.phase_flip(0.01))` inserts
  errors after every circuit moment, on idle qubits too. Pass `qubits=[...]` for
  targeted faults. Triple control is `LineQubit(0)` and systems follow; SIC
  tomography has different register roles. Do not silently equate targeted roles.
- `readout_error`: classical independent final bit flips. The triple circuit has
  one measured bit; SIC tomography has `2*log2(d)` bits. These are different
  confusion maps even at the same bit-error rate.
- `circuit_transform`: compile or otherwise transform each circuit before moment
  noise is inserted. Preserve qubits, omit measurements. For example,
  `lambda c: cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset())`.
  This also transforms any explicit preparation channel. Custom transforms must
  not optimize away intended errors or change outcome conventions.

Default circuits contain ideal dense preparation/POVM gates and CSWAPs. A per-moment
rate on these logical circuits is **not** a calibrated native-gate error rate.
Compilation changes error exposure. The standard sweep is a logical-circuit study;
compare a specified native compilation separately before making device predictions.
Cirq's `depolarize(p)` shrinks a qubit Bloch vector by `1-4*p/3`, not `1-p`.
`phase_flip(p)` shrinks transverse coordinates by `1-2*p`, whereas
`phase_damp(gamma)` uses `sqrt(1-gamma)`. Coherent `rz(theta)` uses radians.

## Metrics and interpretation

`triple_tensor_diagnostics` reports RMSE/max error against a reference, raw
permutation RMS, diagonal/repeated-index/closure violations, center minimum
eigenvalue, and a sufficient emptiness certificate. If every pencil vertex has
negative trace, no simplex combination can be PSD. Failure of this certificate
does not imply nonemptiness. Symmetrization discards register asymmetry; compute
these diagnostics on `raw_tensor`, not just `tensor`.

`compare_state_spaces` explicitly symmetrizes real triples. It intersects PSD
constraints with the probability simplex, samples uniform directions in the full
`d**2-1` dimensional probability hyperplane, and returns radial RMSE, sampled
maximum radial error, relative volume, Jaccard overlap and radius ratios. Euclidean
distances use SIC probability coordinates. Maximum radial error is **not** a
Hausdorff distance. Volume Monte Carlo standard error uses the paired ratio delta
method; it excludes shot uncertainty and is only an asymptotic error estimate.
High-dimensional volume estimation may require many more directions and repeated
seeds. `center_not_strictly_feasible` omits geometric metrics: it is neither zero
volume nor proof of emptiness. Boundary/singular centers are unsupported.

`tomography_cloud_metrics` reports paired total variation, empirical centroid
shift, sorted covariance-axis ratios, triple acceptance and nominal physicality.
It also fits an affine map from known ideal probabilities to observed ones. When
inputs span the whole affine hyperplane it returns singular values and center
shift; for qubits its determinant gives an affine-ball volume ratio. This is a
model-dependent estimate, not a CP certificate. Covariance ratios depend on input
coverage; affine singular values are preferable for anisotropy. Pure Haar inputs
sample the qubit boundary; for d>2 they do not sample the entire mixed-state
boundary. Do not extrapolate a finite-cloud hull to the full body without checking
coverage or assuming a validated linear map.

Qubit convex-hull volume is compared both with the ideal cloud's hull (paired
coverage control) and the true ball. Finite pure-state clouds underestimate the
ideal ball; shot noise can inflate hulls or produce unphysical linear inversions.
No PSD projection is performed, since it would conceal diagnostic evidence.

The binomial bootstrap resamples each independent setting once before symmetry
averaging. Percentile ranges are conditional sampling-error summaries, not
coverage-guaranteed intervals. Endpoints remain deterministic. Track failed
centers explicitly; never silently drop failures when reporting uncertainty.

## Scope, cost and validation

Circuit helpers currently accept power-of-two dimensions >=2 and pure input kets.
This avoids silently postselecting leakage from non-power-of-two embeddings.
Geometry/identity helpers also work with other SIC dimensions. Ordered triples
cost `d**6` circuits; unordered triples cost `choose(d**2+2,3)` but **assume**
permutation symmetry. Dense tensor storage is O(d^6), and cycle-test density
matrices require O(2^(2*(1+3*log2(d)))) memory. `max_settings=4096` guards accidental
large runs; raising it is an explicit resource decision. Tomography holds the
whole cloud in memory. These helpers are not a scalable large-device simulator.

Tests cover exact SIC recovery, Kraus-vs-circuit channel predictions, analytic
volume formulas, amplitude-damping affine geometry, a coherent-unitary blind
spot, ordered asymmetry, readout confusion, seeded shot/bootstrap behavior,
compilation, invalid centers, singular clouds and resource/validation failures.
See [validation](validation.md) and `tests/test_noise.py`.
