"""Complex triple-product measurements and SIC state-space reconstruction.

The convention throughout is ``T[i,j,k] = Tr(Pi_i Pi_j Pi_k)`` for trace-one
projectors. Cyclic permutations leave T unchanged; odd permutations complex
conjugate it. Real triples suffice for the state-space positivity pencil,
while imaginary triples retain the orientation of the operator algebra.

Legacy section/plotting helpers are re-exported from :mod:`cirq_sic.geometry`.
"""

from __future__ import annotations

import cirq
import numpy as np

from .analysis import _density_matrix, _probabilities
from .geometry import (
    make_sum_zero_section,
    section_coordinates,
    probability_spectrahedron_section_2D,
    probability_spectrahedron_section_3D,
    make_icosphere,
    plot_section_2D_matplotlib,
    plot_section_3D_matplotlib,
    mesh_edges_trace,
    plot_section_3D_plotly,
)
from .measurements import prepare_state

__all__ = [
    "triple_products",
    "triple_product_circuit",
    "triple_product_estimate",
    "assemble_triple_products",
    "sic_state_moments",
    "sic_state_constraints",
    "triple_product_state_space",
    "make_sum_zero_section",
    "section_coordinates",
    "probability_spectrahedron_section_2D",
    "probability_spectrahedron_section_3D",
    "make_icosphere",
    "plot_section_2D_matplotlib",
    "plot_section_3D_matplotlib",
    "mesh_edges_trace",
    "plot_section_3D_plotly",
]


def triple_products(states):
    """Return ``Tr(rho_i rho_j rho_k)`` for a sequence of kets or matrices.

    Kets have shape ``(n_states, d)``; density matrices have shape
    ``(n_states, d, d)``. SIC *projectors*, not effects scaled by 1/d,
    produce the tensor required by the SIC reconstruction functions below.
    Dense output has ``n_states**3`` complex entries; avoid allocating full
    tensors for large dimensions and instead select experimental triplets.
    """
    matrices = np.asarray([_density_matrix(state) for state in states])
    if matrices.ndim != 3 or not len(matrices):
        raise ValueError(
            "states must be a nonempty sequence of equal-dimensional states."
        )
    pair_products = np.einsum("iab,jbc->ijac", matrices, matrices)
    return np.einsum("ijac,kca->ijk", pair_products, matrices)


def triple_product_circuit(states, *, component="real", key="triple", measure=True):
    """Build a Hadamard cycle test for three logical states in any dimension.

    ``states`` is exactly three normalized kets or density matrices of the
    same dimension d. Each is embedded in ``n=ceil(log2(d))`` qubits. The
    control is LineQubit(0), followed by the three n-qubit system registers.
    Density matrices add n purification qubits each, after all three system
    registers; these environments are neither swapped nor measured. All
    qubits begin in zero. Thus mixed inputs describe actual preparations,
    with the same ``Tr(rho1 rho2 rho3)`` expectation as the pure-state test.

    Two controlled register swaps implement ``|a,b,c> -> |b,c,a>``. Measuring
    control X returns the real part and control Y the imaginary part;
    ``P(0)-P(1)`` has the requested sign in both cases. Set ``measure=False``
    to return the circuit just before its final computational measurement.
    Dense preparation gates and CSWAPs require compilation for a hardware
    native gateset; this routine does not assume a particular topology.
    """
    if component not in ("real", "imag"):
        raise ValueError("component must be 'real' or 'imag'.")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a nonempty string.")
    states = tuple(np.asarray(state, dtype=complex) for state in states)
    if len(states) != 3:
        raise ValueError("Exactly three states are required.")
    matrices = tuple(_density_matrix(state) for state in states)
    d = len(matrices[0])
    if any(matrix.shape != (d, d) for matrix in matrices):
        raise ValueError("All states must have the same dimension.")
    n = (d - 1).bit_length()
    control = cirq.LineQubit(0)
    registers = [
        tuple(cirq.LineQubit.range(1 + i * n, 1 + (i + 1) * n)) for i in range(3)
    ]
    next_qubit = 1 + 3 * n
    circuit = cirq.Circuit()
    for state, register in zip(states, registers):
        environment = ()
        if state.ndim == 2:
            environment = tuple(cirq.LineQubit.range(next_qubit, next_qubit + n))
            next_qubit += n
        if n:
            circuit.append(
                prepare_state(state, register, environment_qubits=environment)
            )
    circuit.append(cirq.H(control))
    for first, second in ((registers[0], registers[2]), (registers[0], registers[1])):
        circuit.append(cirq.CSWAP(control, a, b) for a, b in zip(first, second))
    if component == "imag":
        circuit.append(cirq.S(control) ** -1)
    circuit.append(cirq.H(control))
    if measure:
        circuit.append(cirq.measure(control, key=key))
    return circuit


def triple_product_estimate(probabilities):
    """Convert binary cycle-test probabilities to ``P(0)-P(1)``.

    Leading batch axes are supported. The same expression applies to the
    real and imaginary tests; combining them is the caller's responsibility.
    For N shots, a plug-in standard error is ``sqrt((1-estimate**2)/N)``.
    """
    probabilities = _probabilities(probabilities)
    if probabilities.shape[-1] != 2:
        raise ValueError("Cycle-test probabilities must have final axis of size two.")
    return probabilities[..., 0] - probabilities[..., 1]


def assemble_triple_products(estimates, triplets, n_states, *, atol=1e-7):
    """Expand complex triple estimates using cyclic/conjugation symmetry.

    Pass one complex estimate per unordered triplet, typically those from
    ``combinations_with_replacement(range(n_states), 3)``. Each triplet's
    supplied order fixes the sign of its imaginary component. Triplets with
    repeated indices are real: their sampled imaginary component is
    discarded because that component is exactly zero. Duplicate estimates
    must agree within ``atol``. Incomplete coverage raises ValueError rather
    than silently replacing unmeasured tensor entries with zeros.
    """
    if (
        isinstance(n_states, (bool, np.bool_))
        or not isinstance(n_states, (int, np.integer))
        or n_states < 1
    ):
        raise ValueError("n_states must be a positive integer.")
    estimates, triplets = list(estimates), list(triplets)
    if len(estimates) != len(triplets):
        raise ValueError("Supply one estimate per triplet.")
    tensor = np.full((n_states,) * 3, np.nan + 1j * np.nan, dtype=complex)
    for estimate, triplet in zip(estimates, triplets):
        if len(triplet) != 3 or any(
            not isinstance(i, (int, np.integer)) or i < 0 or i >= n_states
            for i in triplet
        ):
            raise ValueError("Every triplet must contain three valid state indices.")
        value = complex(estimate)
        if not np.isfinite(value):
            raise ValueError("Triple estimates must be finite.")
        i, j, k = triplet
        if len(set(triplet)) < 3:
            value = complex(value.real)
        for index, entry in (
            ((i, j, k), value),
            ((j, k, i), value),
            ((k, i, j), value),
            ((j, i, k), value.conjugate()),
            ((i, k, j), value.conjugate()),
            ((k, j, i), value.conjugate()),
        ):
            if np.isfinite(tensor[index]) and not np.isclose(
                tensor[index], entry, atol=atol, rtol=0
            ):
                raise ValueError(
                    "Duplicate triple estimates disagree with tensor symmetries."
                )
            tensor[index] = entry
    if np.isnan(tensor).any():
        raise ValueError("Triplets do not cover the complete tensor.")
    return tensor


def _sic_tensor(triples, dimension):
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer.")
    tensor = np.asarray(triples, dtype=complex)
    if tensor.shape != (dimension * dimension,) * 3 or not np.all(np.isfinite(tensor)):
        raise ValueError("Triple tensor must be finite with shape (d**2, d**2, d**2).")
    return tensor


def sic_state_moments(probabilities, triples, dimension):
    """Compute ``Tr(rho²)`` and ``Tr(rho³)`` from SIC probabilities/triples.

    Write ``rho = sum_i a_i Pi_i`` with ``a_i=(d+1)p_i-1/d``. Contract
    ``a_i a_j`` with the analytic SIC Gram matrix and ``a_i a_j a_k`` with
    T. The cubic contraction uses the real tensor component; imaginary
    contributions cancel. The input may be an experimentally estimated
    tensor, so no assertion that it describes an exact SIC is made.
    """
    tensor = _sic_tensor(triples, dimension)
    p = _probabilities(probabilities)
    if p.shape[-1] != dimension**2:
        raise ValueError("SIC probabilities must have final axis d**2.")
    coefficients = (dimension + 1) * p - 1 / dimension
    gram = (np.ones((dimension**2,) * 2) + dimension * np.eye(dimension**2)) / (
        dimension + 1
    )
    purity = np.einsum("...i,...j,ij->...", coefficients, coefficients, gram)
    cubic = np.einsum(
        "...i,...j,...k,ijk->...",
        coefficients,
        coefficients,
        coefficients,
        tensor.real,
        optimize=True,
    )
    return {"purity": purity, "cubic": cubic}


def triple_product_state_space(triples, dimension, *, real=True):
    r"""Reconstruct a positivity pencil from SIC triple measurements alone.

    Return matrices ``A[i]`` with shape ``(d²,d²,d²)`` such that a normalized
    probability vector describes a quantum state iff ``sum_i p_i A[i]`` is
    positive semidefinite, assuming exact SIC triple data. Explicitly,

    ``A[i,j,k] = (d+1) T[j,i,k] - G[j,k]``,

    where ``G[j,k]=Tr(Pi_j Pi_k)=(d*delta_jk+1)/(d+1)``. For any coefficient
    vector x and ``X=sum_k x_k Pi_k``, the resulting quadratic form is
    ``Tr(X† rho X)``. It is nonnegative for all X iff rho is positive.
    With ``real=True``, real x span all Hermitian X and suffice for the same
    test; the real symmetric pencil therefore needs only real triples.

    Measured noise may violate tensor identities. The returned pencil is
    Hermitianized, but this does not fit or certify a valid SIC geometry.
    Its feasible set is an experimental estimate and can even be empty.
    This is reconstruction of the probability state space, not recovery of
    a unique Hilbert-space orientation or SIC fiducial.
    """
    tensor = _sic_tensor(triples, dimension)
    gram = (np.ones((dimension**2,) * 2) + dimension * np.eye(dimension**2)) / (
        dimension + 1
    )
    matrices = (dimension + 1) * tensor.transpose(1, 0, 2) - gram
    matrices = (matrices + matrices.conj().swapaxes(-1, -2)) / 2
    return matrices.real if real else matrices


def sic_state_constraints(probabilities, triples, dimension, *, atol=1e-8):
    """Evaluate moment and complete positivity constraints in SIC coordinates.

    Return purity, cubic moment, their residuals from one, the minimum
    eigenvalue of the triple-derived positivity pencil, and booleans
    ``is_physical``/``is_pure``. The two unit-moment equations characterize
    pure states for exact SIC data, but mixed-state moment bounds alone
    do not characterize positivity in general dimensions. The matrix
    pencil supplies that missing check. Noisy triples yield diagnostics,
    not a certificate about an unknown exact SIC.

    The pure-state moment characterization is discussed in Appleby,
    Ericsson and Fuchs, https://arxiv.org/abs/0910.2750.
    """
    p = _probabilities(probabilities)
    moments = sic_state_moments(p, triples, dimension)
    pencil = triple_product_state_space(triples, dimension)
    matrices = np.einsum("...i,ijk->...jk", p, pencil)
    minimum = np.linalg.eigvalsh(matrices)[..., 0]
    physical = minimum >= -atol
    return {
        **moments,
        "purity_residual": moments["purity"] - 1,
        "cubic_residual": moments["cubic"] - 1,
        "minimum_eigenvalue": minimum,
        "is_physical": physical,
        "is_pure": physical
        & (np.abs(moments["purity"] - 1) <= atol)
        & (np.abs(moments["cubic"] - 1) <= atol),
    }
