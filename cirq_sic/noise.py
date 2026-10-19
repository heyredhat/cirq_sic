"""Circuit noise experiments and geometric diagnostics in SIC coordinates.

Dense small-system tools: no hardware calibration or physical SIC fitting is
implied. Geometry uses an explicitly symmetrized real tensor and the *ideal*
SIC Gram matrix. See docs/noise_state_space.md for assumptions and scaling.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations, product
from math import comb

import cirq
import numpy as np
from scipy.linalg import null_space

from cirq_sic.measurements import sic_povm, build_measurement_circuit
from cirq_sic.triples import (
    triple_product_circuit,
    triple_products,
    triple_product_state_space,
    _sic_tensor,
)

__all__ = [
    "LocalMomentNoise",
    "TripleNoiseResult",
    "simulate_triple_tensor",
    "simulate_sic_tomography",
    "triple_tensor_diagnostics",
    "compare_state_spaces",
    "tomography_cloud_metrics",
]


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rate(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("readout_error must be in [0, 1]")
    return float(value)


def _measurement(dimension):
    dimension = _positive_integer(dimension, "dimension")
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("Circuit studies require a power-of-two dimension >= 2")
    return sic_povm(dimension)


def _channel(channel):
    if channel is not None and (
        not isinstance(channel, cirq.Gate)
        or cirq.num_qubits(channel) != 1
        or not cirq.has_kraus(channel)
    ):
        raise ValueError("channel must be a one-qubit gate with a Kraus representation")


class LocalMomentNoise(cirq.NoiseModel):
    """Apply a one-qubit channel after each nonmeasurement circuit moment.

    ``qubits=None`` includes all circuit qubits, including idle ones. An explicit
    iterable selects qubits; absent qubits are ignored. Channels may also be
    coherent unitary gates. Rates are per *moment*, not per native gate or time.
    Noise applies after preparation and final basis rotation too. Compile the
    circuit with ``circuit_transform`` before applying this model to compare
    gate decompositions. Measurement/readout errors are modeled separately.
    """

    def __init__(self, channel, *, qubits=None):
        _channel(channel)
        if channel is None:
            raise ValueError("channel cannot be None")
        self.channel = channel
        self.qubits = None if qubits is None else frozenset(qubits)

    def noisy_moment(self, moment, system_qubits):
        """Insert a channel moment, preserving original moment boundaries."""
        if any(cirq.is_measurement(op) for op in moment.operations):
            return moment
        targets = [q for q in system_qubits if self.qubits is None or q in self.qubits]
        return [moment, cirq.Moment(self.channel.on(q) for q in targets)]


def _symmetrize(tensor):
    return sum(tensor.transpose(p) for p in permutations(range(3))) / 6


def _assemble(values, triplets, n, ordered):
    tensor = np.empty((n,) * 3)
    for value, triplet in zip(values, triplets):
        indices = [tuple(triplet)] if ordered else set(permutations(triplet))
        for index in indices:
            tensor[index] = value
    return tensor


@dataclass
class TripleNoiseResult:
    """Raw circuit outcomes plus symmetrized reconstruction input.

    ``exact_expectations`` include channel/readout noise but no shot noise.
    ``estimates`` and ``counts_zero`` refer to independent measured settings;
    permutations copied during assembly are *not* independent observations.
    ``standard_errors`` are plug-in binomial errors and can vanish at endpoints.
    ``ordered=False`` imposes permutation symmetry, so cannot test it.
    """

    dimension: int
    triplets: np.ndarray
    exact_expectations: np.ndarray
    estimates: np.ndarray
    counts_zero: object
    shots: object
    ordered: bool

    @property
    def raw_tensor(self):
        """Measured tensor, before averaging permutation-related settings."""
        return _assemble(self.estimates, self.triplets, self.dimension**2, self.ordered)

    @property
    def tensor(self):
        """Fully symmetric real tensor used for reconstruction."""
        return _symmetrize(self.raw_tensor)

    @property
    def standard_errors(self):
        """Binomial plug-in standard errors, or zeros for exact simulation."""
        return (
            np.zeros_like(self.estimates)
            if self.shots is None
            else np.sqrt(np.maximum(0, 1 - self.estimates**2) / self.shots)
        )

    def bootstrap(self, repetitions=100, *, seed=None):
        """Parametric binomial bootstrap tensors, resampling settings once.

        Uses observed probabilities; endpoint settings remain deterministic.
        These are sampling-error replicates, not confidence in a noise model.
        """
        repetitions = _positive_integer(repetitions, "repetitions")
        if self.shots is None:
            raise ValueError("bootstrap requires finite-shot results")
        rng = np.random.default_rng(seed)
        counts = rng.binomial(
            self.shots,
            np.clip((1 + self.estimates) / 2, 0, 1),
            size=(repetitions, len(self.estimates)),
        )
        return np.array(
            [
                _symmetrize(
                    _assemble(
                        2 * c / self.shots - 1,
                        self.triplets,
                        self.dimension**2,
                        self.ordered,
                    )
                )
                for c in counts
            ]
        )


def simulate_triple_tensor(
    dimension=2,
    *,
    noise=None,
    preparation_channel=None,
    readout_error=0,
    shots=None,
    seed=None,
    ordered=True,
    circuit_transform=None,
    max_settings=4096,
):
    """Estimate real SIC triples using density-matrix cycle-test circuits.

    Power-of-two dimensions only. ``preparation_channel`` acts once on each
    system qubit after ideal pure preparation, before the control Hadamard.
    ``noise`` is a Cirq noise model; ``readout_error`` independently flips the
    final control bit. ``shots=None`` gives exact noisy expectations; otherwise
    sample independent binomial counts from those circuit probabilities.
    ``max_settings`` guards accidental large dense runs (default 4096).
    ``ordered=True`` measures d^6 settings to retain register asymmetry;
    False measures combinations with replacement and imposes symmetry.
    ``circuit_transform`` runs after preparation-channel insertion, before
    moment noise. It must preserve qubits and omit measurements.
    """
    dimension = _positive_integer(dimension, "dimension")
    settings = dimension**6 if ordered else comb(dimension**2 + 2, 3)
    if settings > _positive_integer(max_settings, "max_settings"):
        raise ValueError(
            "Study exceeds max_settings; explicitly raise the limit for expensive runs"
        )
    m = _measurement(dimension)
    _channel(preparation_channel)
    readout_error = _rate(readout_error)
    if shots is not None:
        shots = _positive_integer(shots, "shots")
    triplets = np.array(
        list(
            product(range(dimension**2), repeat=3)
            if ordered
            else combinations_with_replacement(range(dimension**2), 3)
        )
    )
    simulator = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)
    control = cirq.LineQubit(0)
    exact = []
    for indices in triplets:
        circuit = triple_product_circuit(m.states[:, indices].T, measure=False)
        if preparation_channel is not None:
            first_h = next(
                i
                for i, moment in enumerate(circuit)
                if cirq.H(control) in moment.operations
            )
            system = sorted(circuit.all_qubits() - {control})
            # Cirq packs the control H alongside disjoint preparations. Split
            # that moment so anisotropic channels act AFTER preparation.
            preparation_moment = cirq.Moment(
                op for op in circuit[first_h].operations if op != cirq.H(control)
            )
            circuit = cirq.Circuit(
                list(circuit[:first_h])
                + [
                    preparation_moment,
                    cirq.Moment(preparation_channel.on(q) for q in system),
                    cirq.Moment([cirq.H(control)]),
                ]
                + list(circuit[first_h + 1 :])
            )
        if circuit_transform is not None:
            circuit = cirq.Circuit(circuit_transform(circuit))
        if circuit.has_measurements():
            raise ValueError("circuit_transform must not add measurements")
        order = [control] + sorted(circuit.all_qubits() - {control})
        rho = simulator.simulate(circuit, qubit_order=order).final_density_matrix
        p0 = np.diag(rho).real.reshape(2, -1)[0].sum()
        exact.append((2 * p0 - 1) * (1 - 2 * readout_error))
    exact = np.clip(exact, -1, 1)
    counts = (
        None
        if shots is None
        else np.random.default_rng(seed).binomial(shots, (1 + exact) / 2)
    )
    estimates = exact.copy() if shots is None else 2 * counts / shots - 1
    return TripleNoiseResult(
        dimension, triplets, exact, estimates, counts, shots, bool(ordered)
    )


def simulate_sic_tomography(
    states,
    *,
    noise=None,
    preparation_channel=None,
    readout_error=0,
    shots=None,
    seed=None,
    circuit_transform=None,
):
    """Simulate pure-state preparation followed by a SIC Naimark measurement.

    Returns ideal, exact noisy, and optionally multinomial-sampled probability
    clouds in the same *nominal* SIC coordinates. Readout flips each measured
    bit independently. Preparation noise acts once on system qubits before
    the measurement unitary; moment noise includes measurement ancillas.
    No positivity projection or renormalization onto a fitted state space is
    performed. Input is a nonempty array of normalized kets, shape (N,d).
    """
    states = np.asarray(states, dtype=complex)
    if (
        states.ndim != 2
        or not len(states)
        or not np.all(np.isfinite(states))
        or not np.allclose(np.linalg.norm(states, axis=1), 1, atol=1e-9, rtol=0)
    ):
        raise ValueError("states must be rows of normalized finite kets")
    m = _measurement(states.shape[1])
    _channel(preparation_channel)
    readout_error = _rate(readout_error)
    if shots is not None:
        shots = _positive_integer(shots, "shots")
    simulator = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)
    confusion = np.array(
        [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    )
    readout = np.array([[1.0]])
    for _ in range(m.num_qubits):
        readout = np.kron(readout, confusion)
    exact = []
    for state in states:
        spec = build_measurement_circuit(m, state, measure=False)
        circuit = spec.circuit.copy()
        if preparation_channel is not None:
            circuit.insert(
                len(circuit) - 1,
                cirq.Moment(preparation_channel.on(q) for q in spec.system_qubits),
            )
        if circuit_transform is not None:
            circuit = cirq.Circuit(circuit_transform(circuit))
        if circuit.has_measurements():
            raise ValueError("circuit_transform must not add measurements")
        rho = simulator.simulate(
            circuit, qubit_order=spec.qubit_order
        ).final_density_matrix
        p = readout @ np.diag(rho).real
        p = np.maximum(p, 0)
        exact.append(p / p.sum())
    exact = np.array(exact)
    rng = np.random.default_rng(seed)
    counts = (
        None if shots is None else np.array([rng.multinomial(shots, p) for p in exact])
    )
    return {
        "ideal": np.array([m.probabilities(s) for s in states]),
        "exact": exact,
        "probabilities": exact.copy() if shots is None else counts / shots,
        "counts": counts,
        "shots": shots,
    }


def triple_tensor_diagnostics(triples, dimension, *, reference=None):
    """Report real-tensor SIC identity violations before symmetrization.

    Closure is sum_i T[i,j,k] = d G[j,k]; repeated-index entries equal G.
    Residuals measure inconsistency with an ideal SIC, not a fitted channel.
    Pass raw ordered data to retain permutation residuals.
    """
    t = _sic_tensor(triples, dimension).real
    n = dimension**2
    gram = (np.ones((n, n)) + dimension * np.eye(n)) / (dimension + 1)
    sym = _symmetrize(t)
    center = np.full(n, 1 / n)
    pencil = triple_product_state_space(sym, dimension)
    result = {
        "permutation_rms": float(np.sqrt(np.mean((t - sym) ** 2))),
        "diagonal_rms": float(
            np.sqrt(np.mean((t[np.arange(n), np.arange(n), np.arange(n)] - 1) ** 2))
        ),
        "repeated_rms": float(
            np.sqrt(np.mean((t[np.arange(n), np.arange(n), :] - gram) ** 2))
        ),
        "closure_rms": float(np.sqrt(np.mean((t.sum(axis=0) - dimension * gram) ** 2))),
        "center_min_eigenvalue": float(
            np.linalg.eigvalsh(np.einsum("i,ijk->jk", center, pencil))[0]
        ),
    }
    traces = np.trace(pencil, axis1=1, axis2=2)
    result["max_vertex_trace"] = float(traces.max())
    result["empty_by_trace_certificate"] = bool(traces.max() < -1e-10)
    if reference is not None:
        ref = _sic_tensor(reference, dimension).real
        result.update(
            tensor_rmse=float(np.sqrt(np.mean((t - ref) ** 2))),
            tensor_max_error=float(np.max(np.abs(t - ref))),
        )
    return result


def _radii(pencil, center, directions, tol):
    origin = np.einsum("i,ijk->jk", center, pencil)
    eigenvalues, vectors = np.linalg.eigh(origin)
    if eigenvalues[0] <= tol:
        return None
    inverse = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    tangent = np.einsum("ri,ijk->rjk", directions, pencil)
    minimum = np.linalg.eigvalsh(inverse @ tangent @ inverse)[:, 0]
    psd = np.divide(
        -1.0, minimum, out=np.full(len(minimum), np.inf), where=minimum < -tol
    )
    simplex = np.min(
        np.divide(
            -center,
            directions,
            out=np.full_like(directions, np.inf),
            where=directions < -tol,
        ),
        axis=1,
    )
    return np.minimum(psd, simplex)


def compare_state_spaces(
    triples, reference, dimension, *, directions=2048, seed=0, atol=1e-10
):
    """Compare full-dimensional radial bodies about the uniform distribution.

    Uses uniform random unit directions in the sum-zero probability hyperplane.
    Distances are Euclidean probability distances. Relative volume is the ratio
    of mean radius**(d**2-1), and Jaccard uses radial intersections/unions.
    ``max_radial_error`` is a sampled radial discrepancy, NOT Hausdorff distance.
    Both centers must be strictly feasible. Otherwise status explicitly reports
    failure and metrics are omitted; an invalid center does not prove emptiness.
    Large-dimensional volume estimates can have very high Monte Carlo variance.
    """
    _positive_integer(directions, "directions")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be positive and finite")
    a = triple_product_state_space(
        _symmetrize(_sic_tensor(triples, dimension).real), dimension
    )
    b = triple_product_state_space(
        _symmetrize(_sic_tensor(reference, dimension).real), dimension
    )
    n = dimension**2
    if n < 2:
        raise ValueError("dimension must be >= 2")
    rng = np.random.default_rng(seed)
    rays = rng.normal(size=(directions, n))
    rays -= rays.mean(axis=1, keepdims=True)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    center = np.full(n, 1 / n)
    r = _radii(a, center, rays, atol)
    s = _radii(b, center, rays, atol)
    if r is None or s is None:
        return {
            "status": "center_not_strictly_feasible",
            "estimate_center_valid": r is not None,
            "reference_center_valid": s is not None,
        }
    power = n - 1
    # Shared scale avoids underflow in powers without changing ratios.
    scale = max(r.max(), s.max())
    rv, sv = (r / scale) ** power, (s / scale) ** power
    ratio = rv.mean() / sv.mean()
    volume_se = (
        np.std(rv - ratio * sv, ddof=1) / np.sqrt(directions) / sv.mean()
        if directions > 1
        else float("nan")
    )
    return {
        "status": "ok",
        "relative_volume_mc_se": float(volume_se),
        "radial_rmse": float(np.sqrt(np.mean((r - s) ** 2))),
        "max_radial_error": float(np.max(abs(r - s))),
        "relative_volume": float(rv.mean() / sv.mean()),
        "jaccard": float(np.minimum(rv, sv).sum() / np.maximum(rv, sv).sum()),
        "relative_radius_min": float(np.min(r / s)),
        "relative_radius_max": float(np.max(r / s)),
        "radii": r,
        "reference_radii": s,
        "directions": rays,
    }


def tomography_cloud_metrics(probabilities, ideal, triples, dimension):
    """Compare paired clouds and their convex hulls in nominal SIC coordinates.

    Reports paired TV error, centroid displacement, covariance axis ratios,
    physicality under ideal linear inversion and triple-pencil acceptance.
    A full-rank paired affine fit reports singular values, center shift and
    residual; it is not a completely-positive-channel certificate.
    For qubits, also reports convex-hull volume ratios in orthonormal probability
    coordinates. Hull volume has finite-cloud bias; it is not a channel volume
    estimate in general. Singular clouds have zero hull volume.
    """
    from scipy.spatial import ConvexHull, QhullError

    p, q = np.asarray(probabilities, float), np.asarray(ideal, float)
    n = dimension**2
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] != n or len(p) < 2:
        raise ValueError("clouds must have matching shape (N,d**2), N >= 2")
    if any(
        not np.all(np.isfinite(x))
        or np.min(x) < -1e-10
        or not np.allclose(x.sum(axis=1), 1, atol=1e-9, rtol=0)
        for x in (p, q)
    ):
        raise ValueError("clouds must contain probability vectors")
    basis = null_space(np.ones((1, n)))
    x, y = p @ basis, q @ basis
    eig_x = np.maximum(np.linalg.eigvalsh(np.atleast_2d(np.cov(x.T))), 0)
    eig_y = np.maximum(np.linalg.eigvalsh(np.atleast_2d(np.cov(y.T))), 0)
    a = triple_product_state_space(
        _symmetrize(_sic_tensor(triples, dimension).real), dimension
    )
    values = np.linalg.eigvalsh(np.einsum("ni,ijk->njk", p, a))[:, 0]
    m = sic_povm(dimension)
    rho = (dimension + 1) * np.einsum("ni,ijk->njk", p, m.effects * dimension) - np.eye(
        dimension
    )
    physical = np.linalg.eigvalsh(rho)[:, 0]
    result = {
        "paired_tv_mean": float(np.mean(np.sum(abs(p - q), axis=1) / 2)),
        "centroid_shift": float(np.linalg.norm(p.mean(axis=0) - q.mean(axis=0))),
        "covariance_axis_ratios": np.sqrt(
            np.divide(
                eig_x, eig_y, out=np.full_like(eig_x, np.nan), where=eig_y > 1e-15
            )
        ),
        "triple_acceptance_fraction": float(np.mean(values >= -1e-8)),
        "nominal_physical_fraction": float(np.mean(physical >= -1e-8)),
        "minimum_nominal_eigenvalue": float(physical.min()),
    }
    design = np.column_stack([y, np.ones(len(y))])
    affine, _, rank, _ = np.linalg.lstsq(design, x, rcond=None)
    result["affine_design_rank"] = int(rank)
    result["affine_fit_rmse"] = float(np.sqrt(np.mean((design @ affine - x) ** 2)))
    if rank == n:
        result["affine_singular_values"] = np.linalg.svd(affine[:-1], compute_uv=False)
        result["affine_center_shift"] = float(np.linalg.norm(affine[-1]))
        # Only for d=2 does a full-rank affine fit to the ball directly describe
        # all ellipsoid axes and volume. A low residual alone does not certify CP.
        if dimension == 2:
            result["affine_ball_volume_ratio"] = float(abs(np.linalg.det(affine[:-1])))
    if dimension == 2:

        def volume(cloud):
            try:
                return float(ConvexHull(cloud).volume)
            except QhullError:
                return 0.0

        vx, vy = volume(x), volume(y)
        ideal_ball_volume = 4 * np.pi / 3 * (1 / np.sqrt(12)) ** 3
        result.update(
            hull_volume=vx,
            ideal_cloud_hull_volume=vy,
            hull_relative_to_ideal_cloud=vx / vy if vy > 0 else float("nan"),
            hull_relative_to_true_body=vx / ideal_ball_volume,
            ideal_cloud_coverage=vy / ideal_ball_volume,
        )
    return result
