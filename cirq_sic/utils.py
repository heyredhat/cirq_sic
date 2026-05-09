from functools import reduce
from pathlib import Path
import re
import numpy as np
import string

####################################################################################

sigma_x = np.array([[0,1], [1,0]])
sigma_y = np.array([[0,-1j], [1j, 0]]) 
sigma_z = np.array([[1,0], [0,-1]])
paulis = [np.eye(2), sigma_x, sigma_y, sigma_z]

def kron(*A):
    """Tensor lots of things together."""
    return reduce(np.kron, A)

def rand_ket(d):
    """Random d-dimensional normalized complex vector."""
    ket = np.random.randn(d) + 1j*np.random.randn(d)
    return ket/np.linalg.norm(ket)

def rand_dm(d, r=1):
    """Random dxd density matrix with rank r."""
    A = np.random.randn(d, r) + 1j*np.random.randn(d, r)
    rho = A @ A.conj().T
    return rho/rho.trace()

def ptrace(rho, over, dims):
    """Partial trace of a density matrix with ket dimensions dims over indices over."""
    indices = list(string.ascii_lowercase[:len(dims)*2])
    for o in over:
        indices[o+len(dims)] = indices[o]
    return np.einsum("".join(indices), rho.reshape(dims*2))

def gram_matrix(E):
    return np.array([[(a@b).trace() for b in E] for a in E])

def bloch_vector(A):
    if len(A.shape) == 1:
        return np.array([A.conj() @ O @ A for O in paulis[1:]]).real
    return np.array([(O @ A).trace() for O in paulis[1:]]).real

####################################################################################

def symmetrize(M, T=100):
    """Obtain a stochastic symmetric matrix by a variant of Sinkhorn's algorithm."""
    for t in range(T):
        M = (M + M.T)/2
        M = M/np.sum(M, axis=0)
    return M

def nonneg_projection(p):
    """Project a vector to the probability simplex by setting negatives to zero and renormalizing."""
    p_fixed = p[:]
    p_fixed[p < 0] = 0
    p_fixed = p_fixed/sum(p_fixed)
    return p_fixed

####################################################################################

def dirac(state_vector):
    """n qubit state vector in Dirac notation."""
    n = int(np.log2(len(state_vector)))
    basis_states = []
    for i, amp in enumerate(state_vector):
        if abs(amp) > 0.001:
            bin_str = format(i, f'0{n}b')
            print("%s: %.2f+%.2fj: %.3f" % (bin_str, amp.real, amp.imag, abs(amp)**2))
            basis_states.append((bin_str, amp))
    return basis_states

####################################################################################

def pad(x, d):
    """Pads a vector with 0's on the right so it is of length d."""
    return np.concatenate([x, np.zeros(int(d-len(x)))])

def mod_d_outcome_mask(d, n, m):
    """When working on computations mod d encoded in n-qubits, with m groups of n-qubits."""
    d_b = 2**n
    return sum([kron(*[np.eye(d_b, dtype=int)[i] for i in ind]) for ind in np.ndindex(*[d_b]*m) if np.all(np.array(ind) < d)])

def mod_d_probabilities(p, d, n, m):
    """Given a probability vector p, for a dimension d computation, which is encoded in m groups of n qubits, return just the relevant probabilities."""
    return p[np.where(mod_d_outcome_mask(d, n, m)==1)]

####################################################################################

def geodesic_interpolator(psi_start, psi_target, atol=1e-12):
    """
    Build psi(t) that moves along the Fubini–Study geodesic from psi_start to psi_target.

    Parameters
    ----------
    psi_start, psi_target : array_like
        Column vectors (shape (d,) or (d,1)). They will be normalized internally.
    atol : float
        Angular tolerance treating states as identical.

    Returns
    -------
    psi : callable
        psi(t) with t ∈ [0, 1], giving a normalized vector.
    """
    psi_start = np.asarray(psi_start, dtype=complex).reshape(-1)
    psi_target = np.asarray(psi_target, dtype=complex).reshape(-1)

    # Normalize both inputs
    psi_start = psi_start / np.linalg.norm(psi_start)
    psi_target = psi_target / np.linalg.norm(psi_target)

    # Align global phase of the target so overlap is real and non-negative
    overlap = np.vdot(psi_start, psi_target)
    phase = np.angle(overlap)
    aligned = np.exp(-1j * phase) * psi_target

    c = np.real(np.vdot(psi_start, aligned))
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < atol:
        def psi(t):
            return psi_start.copy()
        return psi

    sin_theta = np.sin(theta)
    eta = (aligned - c * psi_start) / sin_theta  # unit vector orthogonal to psi_start

    def psi(t):
        t = np.asarray(t, dtype=float)
        return np.cos(t * theta) * psi_start + np.sin(t * theta) * eta

    return psi

####################################################################################

def merge_dicts(dicts):
    """Helper function to merge dictionaries. """
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

####################################################################################

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
    total_height = sum(height for _, height, _ in parsed_svgs) + gap * (len(parsed_svgs) - 1)

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
