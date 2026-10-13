"""Stable dense-unitary synthesis used across supported Cirq versions.

Quantum Shannon decomposition (Shende, Bullock and Markov,
https://arxiv.org/abs/quant-ph/0406176) separates a unitary with cosine-sine
factorization, then demultiplexes block pairs through a unitary eigensystem.
Complex Schur vectors remain orthonormal at repeated eigenvalues, unlike
an unconstrained eigensolver. This matters for identity padding in SIC gates.
"""
import cirq
import numpy as np
from scipy.linalg import cossin, schur


def _multiplex_rotation(qubits, half_angles, rotation):
    """Emit a uniformly controlled rotation using a cyclic Gray-code walk."""
    size = len(half_angles)
    codes = [index ^ (index >> 1) for index in range(size)]
    for index, code in enumerate(codes):
        signs = np.array(
            [(-1) ** bin(code & control).count("1") for control in range(size)]
        )
        angle = 2 * (signs @ half_angles) / size
        yield rotation(float(angle)).on(qubits[0])
        changed_bit = (code ^ codes[(index + 1) % size]).bit_length() - 1
        yield cirq.CNOT(qubits[-1 - changed_bit], qubits[0])


def _demultiplex(qubits, upper, lower):
    """Factor diag(upper, lower) into repeated unitaries and controlled phases."""
    triangular, vectors = schur(upper @ lower.conj().T, output="complex")
    # A unitary matrix is normal, so its Schur form is diagonal up to roundoff.
    phases = np.angle(np.diag(triangular)) / 2
    right = np.exp(1j * phases)[:, None] * (vectors.conj().T @ lower)
    yield from shannon_operations(qubits[1:], right)
    yield from _multiplex_rotation(qubits, -phases, cirq.rz)
    yield from shannon_operations(qubits[1:], vectors)


def shannon_operations(qubits, matrix):
    """Yield one-/two-qubit gates for a unitary, retaining its global phase.

    The caller validates the input operation. Recursive factors use Schur
    decompositions to avoid degeneracy-induced nonunitarity. Two-qubit leaf
    MatrixGates are left for Cirq's target-gateset synthesis.
    """
    if len(qubits) <= 2:
        yield cirq.MatrixGate(matrix).on(*qubits)
        return
    half = matrix.shape[0] // 2
    left, angles, right = cossin(matrix, p=half, q=half, separate=True)
    yield from _demultiplex(qubits, *right)
    yield from _multiplex_rotation(qubits, angles, cirq.ry)
    yield from _demultiplex(qubits, *left)
