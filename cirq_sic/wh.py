"""Weyl–Heisenberg operators with a single explicit outcome convention.

``X|j> = |j+1 mod d>``, ``Z|j> = exp(2πij/d)|j>`` and
``D[a,b] = X**a @ Z**b``. Flattened outcome order is ``k = a*d+b``.
The phase-free displacements are sufficient for all POVM effects.
"""

import numpy as np


def _dimension(d):
    """Validate a positive integer dimension without silently truncating it."""
    if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("dimension must be a positive integer")
    return int(d)


def _fiducial(phi):
    """Validate a finite unit ket; WH completeness depends on its norm."""
    phi = np.asarray(phi, dtype=complex)
    if phi.ndim != 1 or phi.size == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("fiducial must be a finite, nonempty one-dimensional ket")
    if not np.isclose(np.vdot(phi, phi), 1, atol=1e-9, rtol=0):
        raise ValueError("fiducial must have unit norm")
    return phi


def wh_operators(d):
    """Return ``d, w, X, Z, F, D`` for the cyclic WH representation.

    ``F[j,k] = w**(j*k)/sqrt(d)`` and ``X = F† Z F``. The displacement
    array has shape ``(d,d,d,d)``: its first two axes label ``a,b``.
    This dense convenience function uses O(d⁴) storage; :func:`wh_frame`
    constructs the orbit directly using O(d³) storage instead.
    """
    d = _dimension(d)
    w = np.exp(2j * np.pi / d)
    indices = np.arange(d)
    Z = np.diag(w**indices)
    X = np.roll(np.eye(d, dtype=complex), 1, axis=0)
    F = w ** np.outer(indices, indices) / np.sqrt(d)
    D = np.empty((d, d, d, d), dtype=complex)
    for a in range(d):
        for b in range(d):
            D[a, b] = np.roll(np.diag(w ** (b * indices)), a, axis=0)
    return {"d": d, "w": w, "X": X, "Z": Z, "F": F, "D": D}


def wh_frame(phi):
    """Return normalized WH orbit kets as columns of a ``(d,d²)`` frame.

    Column ``a*d+b`` is ``X**a Z**b phi``. These columns have unit norm;
    divide by ``sqrt(d)`` to obtain the weighted POVM frame used by
    :class:`cirq_sic.measurements.Measurement`.
    """
    phi = _fiducial(phi)
    d = len(phi)
    phases = np.exp(2j * np.pi * np.outer(np.arange(d), np.arange(d)) / d)
    clock_orbit = phi[:, None] * phases
    return np.concatenate([np.roll(clock_orbit, a, axis=0) for a in range(d)], axis=1)


def wh_povm(phi):
    """Return WH effects ``|D[a,b] phi><D[a,b] phi|/d``, shape ``(d²,d,d)``.

    This compatibility function returns an array. Use ``wh_measurement``
    when constructing circuits or using the unified experiment workflow.
    """
    frame = wh_frame(phi)
    return np.einsum("ik,jk->kij", frame, frame.conj()) / frame.shape[0]


def change_conjugate_convention(p):
    """Swap ``D Pi D†`` and ``D† Pi D`` outcome conventions.

    The last axis must have square length ``d²``. The map sends ``(a,b)``
    to ``(-a mod d,-b mod d)`` and is its own inverse. Leading batch axes
    are preserved.
    """
    p = np.asarray(p)
    if p.ndim == 0:
        raise ValueError("probabilities must have an outcome axis")
    d = int(np.sqrt(p.shape[-1]))
    if d < 1 or d * d != p.shape[-1]:
        raise ValueError("last axis must have length d²")
    order = (-np.arange(d)) % d
    labels = (order[:, None] * d + order).reshape(-1)
    return p[..., labels]


def arthurs_kelly_ancilla_ready_state(phi, fourier=False):
    """Return the normalized two-qudit Arthurs–Kelly ancilla ready ket.

    ``fourier=True`` additionally applies ``F†`` to both ancillas. Register
    order and conjugate convention agree with the legacy AK circuits.
    """
    phi = _fiducial(phi)
    d = len(phi)
    w = np.exp(2j * np.pi / d)
    F = w ** np.outer(np.arange(d), np.arange(d)) / np.sqrt(d)
    FPi = F.conj().T @ np.outer(phi, phi.conj())
    gamma = np.array([w ** (k * m) * FPi[m, k] for k in range(d) for m in range(d)])
    return np.kron(F.conj().T, F.conj().T) @ gamma if fourier else gamma


__all__ = [
    "wh_operators",
    "wh_frame",
    "wh_povm",
    "change_conjugate_convention",
    "arthurs_kelly_ancilla_ready_state",
]
