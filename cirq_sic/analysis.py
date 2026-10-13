"""Numerical analysis for tomography, WH statistics, and sky/ground tests.

Arrays use outcome-last probability axes. POVMs have shape ``(m, d, d)``;
conditional tables have shape ``(ground_outcomes, reference_outcomes)``.
No simulator, plotting library, or hardware account is needed here.
"""

from __future__ import annotations

from itertools import product

import numpy as np

__all__ = [
    "linear_inversion",
    "project_density_matrix",
    "sic_reconstruction",
    "operator_basis",
    "characteristic_function",
    "stabilizer_distribution",
    "stabilizer_renyi_entropy",
    "entropy_from_probabilities",
    "wh_entropy_from_overlaps",
    "born_rule_prediction",
    "born_rule_diagnostics",
]


def _probabilities(values, *, axis=-1, atol=1e-8):
    """Validate probabilities without silently discarding leakage or negativity."""
    values = np.asarray(values)
    if np.iscomplexobj(values) and np.any(np.abs(values.imag) > atol):
        raise ValueError("Probabilities must be real.")
    values = np.asarray(values.real, dtype=float)
    if values.ndim == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be a finite array.")
    if np.any(values < -atol) or not np.allclose(
        values.sum(axis=axis), 1, atol=atol, rtol=0
    ):
        raise ValueError("Probabilities must be nonnegative and sum to one.")
    return values


def _density_matrix(state, *, atol=1e-8):
    """Validate a normalized ket or density matrix and return its density matrix."""
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        if not len(state) or not np.allclose(
            np.vdot(state, state), 1, atol=atol, rtol=0
        ):
            raise ValueError("A state vector must have norm one.")
        state = np.outer(state, state.conj())
    if state.ndim != 2 or state.shape[0] != state.shape[1] or not state.size:
        raise ValueError("State must be a nonempty ket or square density matrix.")
    if not np.all(np.isfinite(state)) or not np.allclose(
        state, state.conj().T, atol=atol, rtol=0
    ):
        raise ValueError("Density matrix must be finite and Hermitian.")
    if (
        not np.isclose(np.trace(state), 1, atol=atol, rtol=0)
        or np.linalg.eigvalsh(state).min() < -atol
    ):
        raise ValueError(
            "Density matrix must have trace one and be positive semidefinite."
        )
    return state


def _effects(effects, *, atol=1e-8):
    effects = np.asarray(effects, dtype=complex)
    if effects.ndim != 3 or effects.shape[1] != effects.shape[2] or not effects.size:
        raise ValueError(
            "POVM effects must have shape (outcomes, dimension, dimension)."
        )
    if not np.all(np.isfinite(effects)) or not np.allclose(
        effects, effects.conj().swapaxes(-1, -2), atol=atol, rtol=0
    ):
        raise ValueError("POVM effects must be finite and Hermitian.")
    if np.linalg.eigvalsh(effects).min() < -atol:
        raise ValueError("POVM effects must be positive semidefinite.")
    if not np.allclose(
        effects.sum(axis=0), np.eye(effects.shape[1]), atol=atol, rtol=0
    ):
        raise ValueError("POVM effects must sum to identity.")
    return effects


def project_density_matrix(matrix):
    """Project onto trace-one positive matrices in Frobenius distance.

    Hermitianize first, then project its eigenvalues onto the probability
    simplex. This is the Euclidean projection, rather than clipping negative
    eigenvalues and renormalizing. Leading batch axes are supported. This
    is a convenient physical estimate, not a maximum-likelihood fit.
    """
    matrix = np.asarray(matrix, dtype=complex)
    if (
        matrix.ndim < 2
        or matrix.shape[-1] != matrix.shape[-2]
        or not matrix.size
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Expected finite square matrices.")
    values, vectors = np.linalg.eigh((matrix + matrix.conj().swapaxes(-1, -2)) / 2)
    ordered = np.sort(values, axis=-1)[..., ::-1]
    shifts = (np.cumsum(ordered, axis=-1) - 1) / np.arange(1, matrix.shape[-1] + 1)
    count = np.sum(ordered > shifts, axis=-1)
    threshold = np.take_along_axis(shifts, (count - 1)[..., None], axis=-1)
    projected = np.maximum(values - threshold, 0)
    return (vectors * projected[..., None, :]) @ vectors.conj().swapaxes(-1, -2)


def _traceless_basis(d):
    """An orthonormal Hermitian basis excluding normalized identity."""
    matrices = []
    for k in range(1, d):
        diagonal = np.zeros(d)
        diagonal[:k] = 1
        diagonal[k] = -k
        matrices.append(np.diag(diagonal) / np.sqrt(k * (k + 1)))
    for j in range(d):
        for k in range(j + 1, d):
            real = np.zeros((d, d), dtype=complex)
            real[j, k] = real[k, j] = 1 / np.sqrt(2)
            imaginary = np.zeros((d, d), dtype=complex)
            imaginary[j, k], imaginary[k, j] = -1j / np.sqrt(2), 1j / np.sqrt(2)
            matrices.extend((real, imaginary))
    return np.asarray(matrices, dtype=complex).reshape(d * d - 1, d, d)


def linear_inversion(probabilities, effects, *, physical=False, rcond=1e-12):
    """Reconstruct a state from an informationally complete POVM.

    ``probabilities`` has shape ``(..., m)`` and ``effects`` has shape
    ``(m, d, d)``. Solve real least squares in a traceless Hermitian basis,
    enforcing trace one. Rank-deficient POVMs are rejected rather than
    returning an apparently unique state. With finite samples the estimate
    may have negative eigenvalues; ``physical=True`` projects the result
    with :func:`project_density_matrix`. No frequencies are silently
    renormalized, and the input arrays are never modified.
    """
    p, effects = _probabilities(probabilities), _effects(effects)
    m, d, _ = effects.shape
    if p.shape[-1] != m:
        raise ValueError("Probability length must match the number of POVM effects.")
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be positive and finite.")
    basis = _traceless_basis(d)
    design = np.einsum("mij,kji->mk", effects, basis).real
    offset = np.trace(effects, axis1=1, axis2=2).real / d
    observations = (p - offset).reshape(-1, m).T
    coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=rcond)
    if rank < d * d - 1:
        raise ValueError(
            "POVM is not informationally complete at the requested tolerance."
        )
    rho = np.eye(d) / d + np.einsum("kn,kij->nij", coefficients, basis)
    rho = rho.reshape(p.shape[:-1] + (d, d))
    return project_density_matrix(rho) if physical else rho


def sic_reconstruction(probabilities, projectors, *, physical=False, atol=1e-7):
    """Use ``rho = (d+1) sum_i p_i Pi_i - I`` for validated SIC projectors.

    Projectors must be unscaled, with trace one; POVM effects are ``Pi/d``.
    This analytic inverse also works on batches of outcome-last vectors.
    """
    projectors = np.asarray(projectors, dtype=complex)
    if projectors.ndim != 3 or projectors.shape[1] != projectors.shape[2]:
        raise ValueError("Projectors must have shape (d**2, d, d).")
    d = projectors.shape[1]
    _effects(projectors / d, atol=atol)
    gram = np.einsum("aij,bji->ab", projectors, projectors)
    target = (np.ones((d * d, d * d)) + d * np.eye(d * d)) / (d + 1)
    if len(projectors) != d * d or not np.allclose(gram, target, atol=atol, rtol=0):
        raise ValueError("Projectors do not satisfy the SIC overlap conditions.")
    p = _probabilities(probabilities)
    if p.shape[-1] != d * d:
        raise ValueError("A SIC probability vector must have length d**2.")
    rho = (d + 1) * np.einsum("...i,ijk->...jk", p, projectors) - np.eye(d)
    return project_density_matrix(rho) if physical else rho


def operator_basis(dimension, *, basis="pauli"):
    """Return trace-orthogonal unitary operators, with identity first.

    ``basis='pauli'`` means tensor products of I, X, Y, Z in lexicographic
    order and requires ``dimension=2**n``. ``basis='wh'`` means the single
    cyclic-qudit operators ``X**a @ Z**b`` in ``(a,b)`` order for any positive
    dimension. The two choices describe different structures for d >= 4;
    embedding a qudit in qubits does not make them equivalent.
    """
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer.")
    d = int(dimension)
    if basis == "wh":
        from .wh import wh_operators

        return wh_operators(d)["D"].reshape(d * d, d, d)
    if basis != "pauli":
        raise ValueError("basis must be 'pauli' or 'wh'.")
    if d & (d - 1):
        raise ValueError("Tensor Pauli operators require a power-of-two dimension.")
    single = (
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    )
    result = []
    for indices in product(range(4), repeat=d.bit_length() - 1):
        matrix = np.ones((1, 1), dtype=complex)
        for index in indices:
            matrix = np.kron(matrix, single[index])
        result.append(matrix)
    return np.asarray(result)


def characteristic_function(state, *, basis="pauli"):
    """Return ``Tr(rho U)`` in the chosen unitary operator basis."""
    rho = _density_matrix(state)
    return np.einsum("ij,kji->k", rho, operator_basis(len(rho), basis=basis))


def stabilizer_distribution(state, *, basis="pauli"):
    """Return ``Xi_U = |Tr(rho U)|**2 / (d Tr(rho**2))``.

    Operator orthogonality gives ``sum |Tr(rho U)|² = d Tr(rho²)``;
    this therefore remains a probability distribution for mixed states.
    These are characteristic-function weights, not WH-POVM outcome
    probabilities. Use tomography to obtain them from an IC WH-POVM.
    """
    rho = _density_matrix(state)
    weights = np.abs(characteristic_function(rho, basis=basis)) ** 2
    return weights / (len(rho) * np.trace(rho @ rho).real)


def stabilizer_renyi_entropy(state, *, alpha=2, basis="pauli", mixed=False, base=2):
    r"""Compute pure-state stabilizer entropy or its explicit mixed extension.

    For pure states return ``H_alpha(Xi) - log_base(d)``. For mixed states
    explicitly pass ``mixed=True`` to use the purity-corrected extension

    ``H_alpha(Xi) - log_base(d * purity)``,

    where ``Xi_U = |Tr(rho U)|²/(d*purity)``. At alpha=2 this is
    ``-log_base(sum_U |Tr(rho U)|**4 / (d*purity))``. The alpha=1 limit
    uses Shannon entropy. The extension vanishes on maximally mixed states
    and normalized stabilizer-code projectors, but is NOT a faithful magic
    monotone on all mixed states: biased mixtures of |0> and |1> can have
    positive values. For general alpha it is a defined diagnostic, with no
    additional resource-theoretic claim. Pure-state definitions follow
    https://arxiv.org/abs/2106.12587; the mixed alpha=2 expression follows
    https://arxiv.org/abs/2205.02247.

    ``basis='wh'`` computes the analogous cyclic-qudit characteristic
    entropy. It must not be labeled tensor-Pauli magic for encoded qudits.
    Estimating this nonlinear function of reconstructed frequencies has
    finite-shot bias; this function does not correct that bias.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite.")
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one.")
    rho = _density_matrix(state)
    purity = float(np.trace(rho @ rho).real)
    if not mixed and not np.isclose(purity, 1, atol=1e-7, rtol=0):
        raise ValueError(
            "Mixed states require mixed=True to select the purity-corrected extension."
        )
    xi = stabilizer_distribution(rho, basis=basis)
    positive = xi[xi > 0]
    if np.isclose(alpha, 1, atol=1e-10, rtol=0):
        entropy = -np.sum(positive * np.log(positive)) / np.log(base)
    else:
        # Shifted log-sum-exp avoids underflow at large Rényi orders.
        logs = alpha * np.log(positive)
        maximum = logs.max()
        entropy = (maximum + np.log(np.exp(logs - maximum).sum())) / (
            (1 - alpha) * np.log(base)
        )
    result = entropy - np.log(len(rho) * (purity if mixed else 1)) / np.log(base)
    return float(0 if abs(result) < 1e-12 else result)


def entropy_from_probabilities(
    probabilities, effects, *, alpha=2, basis="pauli", mixed=True, base=2, physical=True
):
    """Reconstruct from an IC POVM, then compute the entropy diagnostic.

    A WH-POVM need not be informationally complete for every fiducial;
    :func:`linear_inversion` checks this. Projection is enabled by default
    because sampled linear estimates often have negative eigenvalues.
    Batches return one entropy per probability vector. This is a tomography
    estimator, not an unbiased direct estimate of a fourth-order moment.
    """
    rho = linear_inversion(probabilities, effects, physical=physical)
    d = rho.shape[-1]
    values = [
        stabilizer_renyi_entropy(r, alpha=alpha, basis=basis, mixed=mixed, base=base)
        for r in rho.reshape(-1, d, d)
    ]
    result = np.asarray(values).reshape(rho.shape[:-2])
    return float(result) if result.ndim == 0 else result


def wh_entropy_from_overlaps(probabilities, *, base=2, check_physical=False, atol=1e-8):
    r"""Estimate cyclic-WH M2 directly from two-copy overlap probabilities.

    Supply the outcome-last distribution
    ``s[a,b] = Tr(rho D[a,b] rho D[a,b]†)/d`` in row-major (a,b) order,
    with identity at index zero. The Bell-overlap experiment prepares
    ``rho`` and its complex conjugate to obtain this distribution. These
    are self-overlap probabilities, NOT arbitrary WH-POVM frequencies
    from an unrelated fiducial. Leading batch axes are supported.

    Orthogonality and the discrete symplectic Fourier transform give
    ``sum s² = sum |Tr(rho D)|⁴/d²`` and ``s[0]=Tr(rho²)/d``. Therefore
    the purity-corrected entropy is ``-log_base(sum s²/s[0])``. This is
    the same alpha=2 quantity as :func:`stabilizer_renyi_entropy` with
    ``basis='wh', mixed=True``, including the mixed-state limitations
    documented there. No informationally complete fiducial is required.

    ``check_physical=True`` additionally checks the necessary bounds
    ``1/d² <= s[0] <= 1/d`` and ``s[k] <= s[0]`` to tolerance ``atol``.
    These checks are off by default because finite-shot estimates can
    cross physical endpoints. A zero observed identity probability is
    rejected: its ratio is undefined. Finite-shot estimates have nonlinear
    bias and can be negative; no clipping or bias correction is applied.
    """
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one.")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")
    p = _probabilities(probabilities, atol=atol)
    d = int(np.sqrt(p.shape[-1]))
    if d < 1 or d * d != p.shape[-1]:
        raise ValueError("WH overlap probabilities must have square length d**2.")
    identity_probability = p[..., 0]
    if np.any(identity_probability <= 0):
        raise ValueError("The identity overlap probability must be positive.")
    if check_physical:
        if np.any(identity_probability < 1 / d**2 - atol) or np.any(
            identity_probability > 1 / d + atol
        ):
            raise ValueError("Identity overlap violates the physical purity bounds.")
        if np.any(p > identity_probability[..., None] + atol):
            raise ValueError("No overlap may exceed the identity overlap.")
    result = -np.log(np.sum(p**2, axis=-1) / identity_probability) / np.log(base)
    return float(result) if result.ndim == 0 else result


def _born_transform(sky, conditional, dimension, reference, rcond):
    p, conditional = _probabilities(sky), _probabilities(conditional, axis=0)
    if conditional.ndim != 2 or conditional.shape[1] != p.shape[-1]:
        raise ValueError("conditional must have shape (ground outcomes, sky outcomes).")
    m = p.shape[-1]
    if (dimension is None) == (reference is None):
        raise ValueError(
            "Specify exactly one of dimension (SIC) or reference (calibration)."
        )
    if reference is None:
        if (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or dimension < 1
            or dimension**2 != m
        ):
            raise ValueError("SIC sky probabilities must have length dimension**2.")
        transform = (dimension + 1) * np.eye(m) - np.ones((m, m)) / dimension
    else:
        reference = _probabilities(reference, axis=0)
        if reference.shape != (m, m):
            raise ValueError(
                "reference must be the square reference transition matrix."
            )
        singular_values = np.linalg.svd(reference, compute_uv=False)
        if singular_values[-1] <= rcond * singular_values[0]:
            raise ValueError(
                "Reference matrix is rank deficient at the requested tolerance."
            )
        transform = np.linalg.inv(reference)
    return p, conditional, transform


def born_rule_prediction(
    sky, conditional, *, dimension=None, reference=None, rcond=1e-12
):
    """Predict ground outcomes from sky probabilities and conditional data.

    ``conditional[j,i] = P(ground=j | prepared reference state=i)``.
    For a SIC, provide ``dimension=d`` to evaluate
    ``q_j = sum_i ((d+1)*p_i - 1/d) conditional[j,i]``. For a general
    calibrated IC device, provide ``reference=P`` with
    ``P[i,k]=Tr(E_i sigma_k)`` to evaluate ``C @ inv(P) @ p``.
    This identity assumes the calibration states span operator space;
    rank deficiency is rejected. Negative predictions are returned intact
    as a useful inconsistency diagnostic, never clipped. Batches use the
    last sky axis for outcomes.
    """
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be positive and finite.")
    p, conditional, transform = _born_transform(
        sky, conditional, dimension, reference, rcond
    )
    return np.einsum("ji,...i->...j", conditional @ transform, p)


def born_rule_diagnostics(
    sky,
    conditional,
    ground,
    *,
    dimension=None,
    reference=None,
    sky_shots=None,
    ground_shots=None,
    conditional_shots=None,
    rcond=1e-12,
):
    """Return Born-rule residuals and optional independent-shot uncertainties.

    Outputs include prediction, ``ground-prediction``, L2/max errors, half
    L1 discrepancy, and negative predicted mass. The half L1 quantity is
    a total-variation distance only when both arrays are distributions.
    If sky/ground shot counts are supplied, multinomial covariance is
    propagated through the linear prediction; optional conditional shot
    counts (scalar or one per column) add calibration-table uncertainty.
    Errors assume independent experiments and treat ``reference`` as exact.
    They omit drift, shared calibration noise, and uncertainty in a fitted
    reference inverse. No hypothesis-test p-value is claimed.
    """
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be positive and finite.")
    p, conditional, transform = _born_transform(
        sky, conditional, dimension, reference, rcond
    )
    q = _probabilities(ground)
    prediction = np.einsum("ji,...i->...j", conditional @ transform, p)
    if q.shape != prediction.shape:
        raise ValueError("ground must have the predicted shape.")
    residual = q - prediction
    result = {
        "prediction": prediction,
        "residual": residual,
        "l2_error": np.linalg.norm(residual, axis=-1),
        "max_abs_error": np.max(np.abs(residual), axis=-1),
        "half_l1_error": np.sum(np.abs(residual), axis=-1) / 2,
        "negative_predicted_mass": -np.minimum(prediction, 0).sum(axis=-1),
        "reference_condition": float(np.linalg.cond(transform)),
    }
    if sky_shots is None and ground_shots is None and conditional_shots is None:
        return result
    if sky_shots is None or ground_shots is None:
        raise ValueError(
            "Supply both sky_shots and ground_shots for uncertainty estimates."
        )

    def covariance(probability, shots):
        shots = np.asarray(shots, dtype=float)
        if np.any(~np.isfinite(shots)) or np.any(shots <= 0):
            raise ValueError("Shot counts must be positive and finite.")
        try:
            shots = np.broadcast_to(shots, probability.shape[:-1])
        except ValueError as exc:
            raise ValueError(
                "Shot counts must match probability batch dimensions."
            ) from exc
        return (
            np.einsum("...i,ij->...ij", probability, np.eye(probability.shape[-1]))
            - probability[..., :, None] * probability[..., None, :]
        ) / shots[..., None, None]

    mapping = conditional @ transform
    cov = covariance(q, ground_shots) + np.einsum(
        "ai,...ij,bj->...ab", mapping, covariance(p, sky_shots), mapping
    )
    if conditional_shots is not None:
        weights = np.einsum("ij,...j->...i", transform, p)
        conditional_cov = covariance(conditional.T, conditional_shots)
        cov += np.einsum("...i,iab->...ab", weights**2, conditional_cov)
    result["covariance"] = cov
    result["standard_error"] = np.sqrt(
        np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 0)
    )
    result["standardized_residual"] = np.divide(
        residual,
        result["standard_error"],
        out=np.full_like(residual, np.nan),
        where=result["standard_error"] > 0,
    )
    return result
