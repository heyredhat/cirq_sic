"""Bundled SIC fiducials, design diagnostics, and optional numerical searches.

Fiducials use the cyclic Weyl–Heisenberg convention documented in ``wh``.
Loading this module has no JAX dependency or process-wide configuration
side effects. Search routines use SciPy and accept reproducible seeds.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import binom

from .wh import _dimension, _fiducial, wh_frame, wh_operators


def load_sic_fiducial(d):
    """Load and normalize a bundled dimension-``d`` WH SIC fiducial.

    Data originate from the UMass Boston QBism SIC repository:
    https://www.physics.umb.edu/Research/QBism/solutions.html.
    The real and imaginary components are stored in two text columns.
    Missing dimensions raise ``FileNotFoundError``; use ``sic_povm`` with a
    supplied fiducial to construct measurements in other dimensions.
    """
    d = _dimension(d)
    path = Path(__file__).resolve().parent / "sic_povms" / f"d{d}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No bundled SIC fiducial for dimension {d}; supply your own fiducial."
        )
    data = np.loadtxt(path, ndmin=2)
    if data.shape != (d, 2) or not np.all(np.isfinite(data)):
        raise ValueError(f"Malformed SIC fiducial data in {path.name}")
    fiducial = data[:, 0] + 1j * data[:, 1]
    norm = np.linalg.norm(fiducial)
    if norm == 0:
        raise ValueError(f"Zero SIC fiducial in {path.name}")
    return fiducial / norm


def SIC_P(d):
    """Return SIC outcome probabilities conditioned on each SIC state.

    Rows label measurement outcomes and columns label prepared states:
    ``P[i,j] = (d*delta_ij+1)/(d*(d+1))``.
    """
    d = _dimension(d)
    return (d * np.eye(d * d) + 1) / (d * (d + 1))


def _sic_residual(ket):
    """WH overlap errors via FFT, requiring O(d²) intermediate storage."""
    d = len(ket)
    shifted = np.array([np.roll(ket, -a).conj() * ket for a in range(d)])
    overlaps = np.abs(np.fft.ifft(shifted, axis=1) * d) ** 2
    target = np.full((d, d), 1 / (d + 1))
    target[0, 0] = 1
    return (overlaps - target).reshape(-1)


def refine_sic_fiducial(fiducial, *, tolerance=1e-10, max_nfev=1000):
    """Refine a near-SIC ket using nonlinear WH overlap equations.

    Returns a normalized ket or raises ``ValueError`` if the requested
    maximum squared-overlap error is not reached. This is local numerical
    optimization, not a proof or a general SIC-existence algorithm.
    ``load_sic_fiducial`` always returns the unmodified normalized data.
    """
    fiducial = _fiducial(fiducial)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    max_nfev = _dimension(max_nfev)
    d = len(fiducial)
    if np.max(np.abs(_sic_residual(fiducial))) <= tolerance:
        return fiducial.copy()

    def to_ket(params):
        ket = params[:d] + 1j * params[d:]
        return ket / np.linalg.norm(ket)

    result = least_squares(
        lambda params: _sic_residual(to_ket(params)),
        np.concatenate([fiducial.real, fiducial.imag]),
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=max_nfev,
    )
    refined = to_ket(result.x)
    error = np.max(np.abs(_sic_residual(refined)))
    if error > tolerance:
        raise ValueError(
            f"SIC refinement did not converge: maximum overlap error {error:.3g}"
        )
    return refined


@lru_cache(maxsize=32)
def _measurement_fiducial(d):
    """Accurate small-d measurement fiducials, while preserving raw loaders."""
    fiducial = load_sic_fiducial(d)
    if d <= 32:
        fiducial = refine_sic_fiducial(fiducial)
    fiducial.setflags(write=False)
    return fiducial


def frame_potential_minimum(d, t=2):
    """Welch lower bound ``1/binom(d+t-1,t)`` for normalized t-frame potential."""
    d, t = _dimension(d), _dimension(t)
    return float(1 / binom(d + t - 1, t))


def minimum_design_elements(d, t):
    """Return the projective complex t-design cardinality lower bound."""
    d, t = _dimension(d), _dimension(t)
    lower, upper = t // 2, (t + 1) // 2
    return int(round(binom(d - 1 + lower, lower) * binom(d - 1 + upper, upper)))


def minimize_wh_frame_potential(d, T=10, method="SLSQP", *, seed=None, maxiter=10000):
    """Search for a WH SIC fiducial using ``T`` SciPy optimization restarts.

    This numerical search has no guarantee of finding a SIC. Validate its
    result with ``sic_povm(fiducial=result)`` before using SIC identities.
    Each trial optimizes the real and imaginary components of a normalized
    ket; ``seed`` controls the restart generator. The legacy arguments
    ``d, T, method`` are retained, and JAX is no longer required.
    """
    d, T = _dimension(d), _dimension(T)
    maxiter = _dimension(maxiter)
    if d == 1:
        return np.ones(1, dtype=complex)
    generator = np.random.default_rng(seed)
    minimum = frame_potential_minimum(d)

    def ket_from_params(params):
        ket = params[:d] + 1j * params[d:]
        norm = np.linalg.norm(ket)
        return ket / norm if norm > 0 else np.eye(d, dtype=complex)[0]

    def objective(params):
        ket = ket_from_params(params)
        return np.sum(np.abs(ket.conj() @ wh_frame(ket)) ** 4) / d**2 - minimum

    results = [
        minimize(
            objective,
            generator.standard_normal(2 * d),
            method=method,
            tol=1e-12,
            options={"maxiter": maxiter},
        )
        for _ in range(T)
    ]
    best = min(results, key=lambda result: result.fun)
    return ket_from_params(best.x)


def wh_2_frame_potential(ket):
    """Return squared excess of the WH 2-frame potential over its Welch bound.

    This retains the historical objective's semantics (squared residual,
    not the raw frame potential). The input ket must be normalized.
    """
    ket = _fiducial(ket)
    potential = np.sum(np.abs(ket.conj() @ wh_frame(ket)) ** 4) / len(ket) ** 2
    return float((potential - frame_potential_minimum(len(ket), 2)) ** 2)


def frame_potential(R, t):
    """Return a weighted t-frame potential for the columns of ``R``.

    Columns are normalized internally. Their squared original norms give
    the weights, rescaled to sum to one. Thus an equal-norm frame gives
    ``sum_ij |<psi_i|psi_j>|**(2*t) / m**2``. Zero columns are rejected.
    """
    t = _dimension(t)
    R = np.asarray(R, dtype=complex)
    if R.ndim != 2 or min(R.shape) == 0 or not np.all(np.isfinite(R)):
        raise ValueError("R must be a finite, nonempty two-dimensional frame")
    norms = np.linalg.norm(R, axis=0)
    if np.any(norms == 0):
        raise ValueError("frame columns must be nonzero")
    weights = norms**2 / np.sum(norms**2)
    normalized = R / norms
    return float(
        np.sum(
            np.outer(weights, weights)
            * np.abs(normalized.conj().T @ normalized) ** (2 * t)
        )
    )


def d4_sic_fiducial_ket(monomial=False):
    """Return the analytic d=4 SIC fiducial in cyclic WH or monomial basis."""
    fiducial = np.array([np.sqrt(2 + np.sqrt(5)), 1, 1, 1]) / np.sqrt(5 + np.sqrt(5))
    if monomial:
        return fiducial
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    P = np.diag(np.exp(1j * np.pi * np.array([0, -1 / 4, 1 / 4, 1 / 2])))
    return np.kron(H, np.eye(2)) @ P @ fiducial


def all_d2_fiducials():
    """Return eight qubit SIC kets forming two antipodal Bloch tetrahedra."""
    D = wh_operators(2)["D"].reshape(-1, 2, 2)
    phi = np.array(
        [np.sqrt(3 + np.sqrt(3)), np.exp(1j * np.pi / 4) * np.sqrt(3 - np.sqrt(3))]
    ) / np.sqrt(6)
    # The orthogonal ket has the opposite Bloch vector.
    antipodal = np.array([-phi[1].conj(), phi[0].conj()])
    return np.array(
        [operator @ fiducial for fiducial in (phi, antipodal) for operator in D]
    )


__all__ = [
    "load_sic_fiducial",
    "SIC_P",
    "frame_potential_minimum",
    "minimum_design_elements",
    "minimize_wh_frame_potential",
    "wh_2_frame_potential",
    "frame_potential",
    "d4_sic_fiducial_ket",
    "all_d2_fiducials",
    "refine_sic_fiducial",
]
