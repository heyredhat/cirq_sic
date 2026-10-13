"""Small numerical and presentation helpers shared by the library.

Tensor factors and computational basis indices use big-endian order throughout.
Random-state helpers accept a NumPy generator to make experiments reproducible.
"""

from functools import reduce
from operator import index
from pathlib import Path
import re

import numpy as np

__all__ = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "paulis",
    "kron",
    "rand_ket",
    "rand_dm",
    "ptrace",
    "gram_matrix",
    "bloch_vector",
    "symmetrize",
    "nonneg_projection",
    "dirac",
    "pad",
    "mod_d_outcome_mask",
    "mod_d_probabilities",
    "geodesic_interpolator",
    "merge_dicts",
    "save_svg_circuit",
]

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
paulis = [np.eye(2), sigma_x, sigma_y, sigma_z]


def _positive_integer(value, name):
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def kron(*arrays):
    """Return the Kronecker product of one or more arrays."""
    if not arrays:
        raise ValueError("At least one tensor factor is required.")
    return reduce(np.kron, arrays)


def rand_ket(d, rng=None):
    """Draw a Haar-random normalized complex ket of dimension ``d``.

    ``rng`` may be a ``numpy.random.Generator`` or ``RandomState``. Omitting it
    preserves compatibility with ``numpy.random.seed``.
    """
    d = _positive_integer(d, "d")
    rng = np.random if rng is None else rng
    ket = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return ket / np.linalg.norm(ket)


def rand_dm(d, r=1, rng=None):
    """Draw a complex Wishart density matrix of dimension ``d`` and rank ``r``.

    The rank is ``r`` almost surely; it must lie between one and ``d``. Passing
    a NumPy random generator makes the draw reproducible.
    """
    d, r = _positive_integer(d, "d"), _positive_integer(r, "r")
    if r > d:
        raise ValueError("r must not exceed d.")
    rng = np.random if rng is None else rng
    matrix = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def ptrace(rho, over, dims):
    """Trace out subsystem indices ``over`` from a density matrix.

    ``dims`` gives the dimensions of the tensor factors in big-endian order.
    Remaining factors keep their original order. The result is a two-dimensional
    matrix, including a ``(1, 1)`` matrix when all factors are traced out.
    """
    dims = tuple(_positive_integer(d, "subsystem dimension") for d in dims)
    if not dims:
        raise ValueError("dims must contain at least one subsystem.")
    over = tuple(index(i) for i in over)
    if len(set(over)) != len(over) or any(i < 0 or i >= len(dims) for i in over):
        raise ValueError("over must contain distinct valid subsystem indices.")
    rho = np.asarray(rho)
    dimension = int(np.prod(dims))
    if rho.shape != (dimension, dimension):
        raise ValueError("rho shape does not match the product of dims.")
    reduced = rho.reshape(dims + dims)
    remaining_dims = list(dims)
    for subsystem in sorted(over, reverse=True):
        reduced = np.trace(
            reduced, axis1=subsystem, axis2=subsystem + len(remaining_dims)
        )
        remaining_dims.pop(subsystem)
    remaining = int(np.prod(remaining_dims))
    return reduced.reshape(remaining, remaining)


def gram_matrix(effects):
    """Return ``G[i,j] = trace(effects[i] @ effects[j])``.

    For Hermitian POVM effects this is the Hilbert--Schmidt Gram matrix.
    """
    effects = np.asarray(effects)
    if effects.ndim != 3 or effects.shape[1] != effects.shape[2]:
        raise ValueError("effects must have shape (n, d, d).")
    return np.einsum("aij,bji->ab", effects, effects)


def bloch_vector(state):
    """Return the three Pauli expectation values of a qubit ket or matrix."""
    state = np.asarray(state)
    if state.shape == (2,):
        return np.array(
            [np.vdot(state, operator @ state) for operator in paulis[1:]]
        ).real
    if state.shape == (2, 2):
        return np.array([np.trace(operator @ state) for operator in paulis[1:]]).real
    raise ValueError(
        "state must be a qubit ket of shape (2,) or matrix of shape (2, 2)."
    )


def symmetrize(matrix, T=100):
    """Iteratively symmetrize and column-normalize a nonnegative square matrix.

    Columns sum to one after each iteration; symmetry is approximate at finite
    ``T``. This routine does not modify its input.
    """
    matrix = np.array(matrix, dtype=float, copy=True)
    T = _positive_integer(T, "T")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("matrix must be nonempty and square.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("matrix must be finite and nonnegative.")
    for _ in range(T):
        matrix = (matrix + matrix.T) / 2
        totals = matrix.sum(axis=0)
        if np.any(totals == 0):
            raise ValueError("Cannot normalize a zero column.")
        matrix /= totals
    return matrix


def nonneg_projection(probabilities):
    """Clip negative entries and normalize a vector without changing the input.

    This is a clipping heuristic, not Euclidean projection onto the simplex.
    An input with no positive mass raises ``ValueError`` instead of producing NaN.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a nonempty finite vector.")
    clipped = np.maximum(values, 0)
    total = clipped.sum()
    if total == 0:
        raise ValueError("probabilities must contain positive mass.")
    return clipped / total


def dirac(state_vector):
    """Print amplitudes above 0.001 and return their bit strings and values."""
    vector = np.asarray(state_vector)
    if vector.ndim != 1 or not vector.size or vector.size & (vector.size - 1):
        raise ValueError("state_vector length must be a positive power of two.")
    n = vector.size.bit_length() - 1
    basis_states = []
    for i, amplitude in enumerate(vector):
        if abs(amplitude) > 0.001:
            bits = format(i, f"0{n}b")
            print(
                f"{bits}: {amplitude.real:.2f}{amplitude.imag:+.2f}j: {abs(amplitude)**2:.3f}"
            )
            basis_states.append((bits, amplitude))
    return basis_states


def pad(vector, d):
    """Right-pad a vector with zeros to length ``d``, preserving its dtype."""
    vector = np.asarray(vector)
    d = _positive_integer(d, "d")
    if vector.ndim != 1 or vector.size > d:
        raise ValueError("vector must be one dimensional and no longer than d.")
    return np.pad(vector, (0, d - vector.size))


def mod_d_outcome_mask(d, n, m):
    """Mark valid outcomes of ``m`` registers of ``n`` qubits encoding dimension ``d``.

    The returned flat integer mask follows big-endian register order. Its cost is
    linear in the output size, rather than enumerating every valid tensor product.
    """
    d = _positive_integer(d, "d")
    n, m = _positive_integer(n, "n"), _positive_integer(m, "m")
    capacity = 2**n
    if d > capacity:
        raise ValueError("d must fit in n qubits.")
    register_mask = (np.arange(capacity) < d).astype(int)
    return kron(*([register_mask] * m))


def mod_d_probabilities(probabilities, d, n, m):
    """Select encoded outcomes, leaving any missing probability mass explicit.

    No renormalization is performed: leaked mass remains observable as a sum
    below one. ``probabilities`` must be a vector of length ``2**(n*m)``.
    """
    probabilities = np.asarray(probabilities)
    mask = mod_d_outcome_mask(d, n, m).astype(bool)
    if probabilities.shape != mask.shape:
        raise ValueError("probabilities shape does not match the encoded registers.")
    return probabilities[mask]


def geodesic_interpolator(psi_start, psi_target, atol=1e-12):
    """Build a normalized Fubini--Study geodesic between two nonzero kets.

    The target's global phase is aligned with the start. The callable accepts a
    scalar or array ``t`` in ``[0, 1]``; array inputs return shape ``t.shape + (d,)``.
    Orthogonal endpoints use the phase of the supplied target.
    """
    start = np.asarray(psi_start, dtype=complex).reshape(-1)
    target = np.asarray(psi_target, dtype=complex).reshape(-1)
    if start.shape != target.shape or not start.size:
        raise ValueError("The endpoint kets must have the same nonzero dimension.")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("The endpoint kets must be finite.")
    if np.linalg.norm(start) == 0 or np.linalg.norm(target) == 0:
        raise ValueError("The endpoint kets must be nonzero.")
    if atol < 0:
        raise ValueError("atol must be nonnegative.")
    start = start / np.linalg.norm(start)
    target = target / np.linalg.norm(target)
    overlap = np.vdot(start, target)
    aligned = np.exp(-1j * np.angle(overlap)) * target
    cosine = np.clip(abs(overlap), 0.0, 1.0)
    theta = np.arccos(cosine)
    # Use the orthogonal residual norm; arccos rounds tiny angles inaccurately.
    residual = aligned - cosine * start
    norm = np.linalg.norm(residual)
    tangent = residual / norm if norm > atol else np.zeros_like(start)

    def interpolate(t):
        t = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t)) or np.any((t < 0) | (t > 1)):
            raise ValueError("t must be finite and lie in [0, 1].")
        if norm <= atol:
            return np.broadcast_to(start, t.shape + start.shape).copy()
        result = (
            np.cos(t[..., None] * theta) * start
            + np.sin(t[..., None] * theta) * tangent
        )
        return result / np.linalg.norm(result, axis=-1, keepdims=True)

    return interpolate


def merge_dicts(dictionaries):
    """Merge dictionaries into a new mapping, with later entries taking priority."""
    merged = {}
    for dictionary in dictionaries:
        merged.update(dictionary)
    return merged


def _chunk_circuit_by_moments(circuit, max_moments_per_chunk):
    """Split a Cirq circuit into contiguous moment chunks."""
    if max_moments_per_chunk is None:
        return [circuit]
    if max_moments_per_chunk <= 0:
        raise ValueError("max_moments_per_chunk must be positive or None.")

    return [
        circuit[start : start + max_moments_per_chunk]
        for start in range(0, len(circuit), max_moments_per_chunk)
    ]


def _stack_svgs(svg_texts, gap=24):
    """Stack SVG images vertically into a single SVG canvas."""
    if not svg_texts:
        raise ValueError("svg_texts must contain at least one SVG.")

    parsed_svgs = []
    pattern = re.compile(
        r"<svg[^>]*width=\"([^\"]+)\"[^>]*height=\"([^\"]+)\"[^>]*>(.*)</svg>",
        re.DOTALL,
    )
    for svg_text in svg_texts:
        match = pattern.fullmatch(svg_text.strip())
        if match is None:
            raise ValueError("Could not parse SVG width/height while stacking panels.")
        width, height, inner = match.groups()
        parsed_svgs.append((float(width), float(height), inner))

    total_width = max(width for width, _, _ in parsed_svgs)
    total_height = sum(height for _, height, _ in parsed_svgs) + gap * (
        len(parsed_svgs) - 1
    )

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}">'
    ]
    y_offset = 0.0
    for width, height, inner in parsed_svgs:
        x_offset = (total_width - width) / 2
        pieces.append(f'<g transform="translate({x_offset},{y_offset})">{inner}</g>')
        y_offset += height + gap
    pieces.append("</svg>")
    return "".join(pieces)


def save_svg_circuit(
    circuit,
    path,
    max_moments_per_panel=None,
    panel_gap=24,
    transpose=False,
):
    """
    Save a Cirq circuit diagram to disk as SVG.

    When ``max_moments_per_panel`` is set, long circuits are split into multiple
    contiguous moment ranges and stacked vertically into one SVG file.
    """
    from cirq.contrib.svg import circuit_to_svg
    from cirq.contrib.svg.svg import tdd_to_svg

    path = Path(path)

    if max_moments_per_panel is None and not transpose:
        svg = circuit_to_svg(circuit)
    else:
        panels = []
        for chunk in _chunk_circuit_by_moments(circuit, max_moments_per_panel):
            if transpose:
                tdd = chunk.to_text_diagram_drawer(transpose=True)
                svg = tdd_to_svg(tdd)
            else:
                svg = circuit_to_svg(chunk)
            if svg:
                panels.append(svg)

        if not panels:
            raise ValueError("Can't draw SVG diagram for an empty circuit.")
        svg = panels[0] if len(panels) == 1 else _stack_svgs(panels, gap=panel_gap)

    path.write_text(svg, encoding="utf-8")
    return path
