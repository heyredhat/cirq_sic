"""Gray-code state-preparation angles and unitary reconstruction.

These helpers operate on power-of-two, normalized kets. The higher-level
``circuits.ansatz_circuit`` pads arbitrary logical dimensions automatically.
Angles are radians, and qubit/register order is big-endian.
"""

from functools import reduce
from itertools import product

import numpy as np
from scipy.linalg import block_diag, expm

from .utils import sigma_y, sigma_z


def _positive_qubit_count(n):
    """Validate a positive number of qubits."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("number of qubits must be a positive integer")
    return int(n)


def grey(n):
    """Return all n-bit reflected Gray words, including one empty word for n=0."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    words = np.arange(2**n)
    codes = words ^ (words >> 1)
    return ((codes[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(int)


def grey_data(k):
    """Return the rotation transform and cyclic CNOT targets for k controls.

    For zero controls return ``None``, since no controlled decomposition is
    required. Otherwise ``M @ angles`` gives Gray-code rotation angles.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0:
        return None
    binary = np.array(list(product([0, 1], repeat=k)))
    gray = grey(k)
    targets = np.argmax(gray != np.roll(gray, -1, axis=0), axis=1).tolist()
    transform = (-1.0) ** (gray @ binary.T) / 2**k
    return transform, targets


def ansatz_params_to_angles(n, params, sign=1):
    """Unpack ``1+2*(2**n-1)`` real preparation parameters.

    The first entry is the global phase, followed by all polar angles and
    then all azimuthal angles, each grouped by levels of size 1,2,4,... .
    ``sign=-1`` conjugates the azimuthal rotations.
    """
    n = _positive_qubit_count(n)
    params = np.asarray(params)
    if np.iscomplexobj(params) and np.any(params.imag != 0):
        raise ValueError("params must be real")
    params = np.asarray(params.real, dtype=float)
    count = 2**n - 1
    if params.shape != (1 + 2 * count,) or not np.all(np.isfinite(params)):
        raise ValueError(f"params must be a finite vector of length {1+2*count}")
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    splits = np.cumsum([2**i for i in range(n - 1)])
    thetas = np.split(params[1 : 1 + count], splits)
    phis = np.split(sign * params[1 + count :], splits)
    return thetas, phis, float(params[0])


def ansatz_angles_to_params(thetas, phis, phase):
    """Flatten matching angle levels and a global phase into parameters."""
    if len(thetas) == 0 or len(thetas) != len(phis):
        raise ValueError("thetas and phis require matching, nonempty levels")
    for index, (theta, phi) in enumerate(zip(thetas, phis)):
        if np.shape(theta) != (2**index,) or np.shape(phi) != (2**index,):
            raise ValueError("angle level i must have length 2**i")
    params = np.concatenate([[phase], np.concatenate(thetas), np.concatenate(phis)])
    if np.iscomplexobj(params) and np.any(params.imag != 0):
        raise ValueError("angles and phase must be real")
    if not np.all(np.isfinite(params)):
        raise ValueError("angles and phase must be finite")
    return params.real


def ket_to_ansatz_angles(ket):
    """Convert a normalized power-of-two ket to preparation angle levels.

    Zero-amplitude branches use zero polar/azimuthal angles, avoiding the
    former ill-conditioned normalization of tiny branches. Overall phase
    is returned separately and preserved by circuit reconstruction.
    """
    ket = np.asarray(ket, dtype=complex)
    if ket.ndim != 1 or len(ket) < 2 or len(ket) & (len(ket) - 1):
        raise ValueError("ket length must be a power of two, at least two")
    if not np.all(np.isfinite(ket)) or not np.isclose(
        np.vdot(ket, ket), 1, atol=1e-9, rtol=0
    ):
        raise ValueError("ket must be finite and normalized")
    current, thetas, phis = ket.copy(), [], []
    for _ in range((len(ket) - 1).bit_length()):
        current_thetas, current_phis, reduced = [], [], []
        for index in range(0, len(current), 2):
            pair = current[index : index + 2]
            theta = float(2 * np.arctan2(abs(pair[1]), abs(pair[0])))
            phi = float(np.angle(pair[1]) - np.angle(pair[0]))
            if not np.any(pair):
                theta = phi = 0.0
            rotation = expm(-1j * sigma_z * phi / 2) @ expm(-1j * sigma_y * theta / 2)
            reduced.append((rotation.conj().T @ pair)[0])
            current_thetas.append(theta)
            current_phis.append(phi)
        current = np.asarray(reduced)
        thetas.append(np.asarray(current_thetas))
        phis.append(np.asarray(current_phis))
    return thetas[::-1], phis[::-1], float(np.angle(current[0]))


def ansatz_unitary(n, params):
    """Reconstruct the dense state-preparation unitary for n qubits."""
    n = _positive_qubit_count(n)
    thetas, phis, phase = ansatz_params_to_angles(n, params)
    polar, azimuthal = thetas[::-1], phis[::-1]
    layers = [
        np.kron(
            block_diag(
                *[
                    expm(-1j * sigma_z * azimuthal[i][j] / 2)
                    @ expm(-1j * sigma_y * polar[i][j] / 2)
                    for j in range(len(polar[i]))
                ]
            ),
            np.eye(2**i),
        )
        for i in range(n)
    ]
    layers.append(np.kron(expm(1j * sigma_z * phase), np.eye(2 ** (n - 1))))
    return reduce(np.matmul, layers)


__all__ = [
    "grey",
    "grey_data",
    "ansatz_params_to_angles",
    "ansatz_angles_to_params",
    "ket_to_ansatz_angles",
    "ansatz_unitary",
]
