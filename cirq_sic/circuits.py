"""Cirq building blocks and compatibility circuits for SIC/WH experiments.

Registers are big-endian. Embedded ``*_d`` operations act exactly on the
first ``d`` basis states and as identity on padding; their historical
``aux`` argument is accepted but auxiliary qubits are no longer needed.
For new code prefer ``measurements.build_measurement_circuit``.
"""

import cirq
import numpy as np
from scipy.linalg import block_diag

from .wh import _dimension
from .ansatz import (
    grey_data,
    ansatz_params_to_angles,
    ansatz_angles_to_params,
    ket_to_ansatz_angles,
)


def measure_register(qubits, prefix):
    """Measure a register one qubit at a time using keys like ``prefix_0``."""
    return cirq.Moment(
        cirq.measure(qubit, key=f"{prefix}_{i}") for i, qubit in enumerate(qubits)
    )


def qudit_basis_state(qubits, m):
    """Prepares the qudit basis state |m> on the qubits q."""
    n = len(qubits)
    if (
        isinstance(m, (bool, np.bool_))
        or not isinstance(m, (int, np.integer))
        or not 0 <= m < 2**n
    ):
        raise ValueError("basis index must fit the register")
    bitstr = bin(m)[2:].zfill(n)
    for i, b in enumerate(bitstr):
        if b == "1":
            yield cirq.X(qubits[i])


def qft(qubits, inverse=False):
    """Qudit Fourier transform on n qubits."""
    if inverse:
        yield from cirq.inverse(list(qft(qubits)))
        return
    n = len(qubits)
    for i in range(n):
        yield cirq.H(qubits[i])
        for j in range(i + 1, n):
            yield cirq.CZPowGate(exponent=2 ** (i - j))(qubits[j], qubits[i])
    for i in range(n // 2):
        yield cirq.SWAP(qubits[i], qubits[n - 1 - i])


def mqft(qubits, inverse=False, key_fn=str):
    """Semiclassical QFT followed immediately by single-qubit measurements.

    For ``inverse=True`` this matches ``qft(qubits, inverse=True)`` followed by
    a computational basis measurement. For ``inverse=False`` it implements the
    no-swap QFT convention.
    """
    n = len(qubits)
    if inverse:
        for i in range(n // 2):
            yield cirq.SWAP(qubits[i], qubits[n - 1 - i])
    indices = range(n) if not inverse else range(n - 1, -1, -1)

    for measured in indices:
        yield cirq.H(qubits[measured])
        key = key_fn(qubits[measured])
        yield cirq.measure(qubits[measured], key=key)

        if inverse:
            remaining = range(measured)
        else:
            remaining = range(measured + 1, n)

        for target in remaining:
            exponent = 2 ** (min(measured, target) - max(measured, target))
            if inverse:
                exponent *= -1
            yield cirq.ZPowGate(exponent=exponent)(
                qubits[target]
            ).with_classical_controls(key)


def Z(qubits, k=1):
    """Apply the clock Z**k in dimension 2**len(qubits)."""
    for j, qubit in enumerate(qubits):
        angle = k * np.pi / (2**j)
        if angle != 0:
            yield cirq.ZPowGate(exponent=angle / np.pi)(qubits[j])


def X(qubits, k=1):
    """Qudit shift on n qubits."""
    yield from qft(qubits)
    yield from Z(qubits, k=k)
    yield from qft(qubits, inverse=True)


def displace(qubits, a1, a2):
    """Qudit displacement operator on n qudits with indices (a1, a2)."""
    yield from Z(qubits, a2)
    yield from X(qubits, a1)


def wh_state(qubits, prepare_fiducial, a1, a2):
    """Prepare the WH state D(a1,a2)|fiducial> on n qubits."""
    yield from prepare_fiducial(qubits)
    yield from displace(qubits, a1, a2)


def QCZ(control_qubit, target_qubits, k=1):
    """Qubit controlled clock on n qubits."""
    for j, target_qubit in enumerate(target_qubits):
        angle = k * np.pi / (2**j)
        if angle != 0:
            yield cirq.CZPowGate(exponent=angle / np.pi)(control_qubit, target_qubit)


def CZ(control_qubits, target_qubits, inverse=False):
    """Qudit controlled clock on n control qubit and n target qubits."""
    if inverse:
        yield from cirq.inverse(list(CZ(control_qubits, target_qubits)))
        return
    n = len(control_qubits)
    for j in range(n):
        yield from QCZ(control_qubits[n - j - 1], target_qubits, 2**j)


def CX(control_qubits, target_qubits, inverse=False):
    """Qudit controlled shift on n control qubits and n target qubits."""
    yield from qft(target_qubits)
    yield from CZ(control_qubits, target_qubits, inverse=inverse)
    yield from qft(target_qubits, inverse=True)


####################################################################################


def ready_arthurs_kelly_ancillas(ancilla1_qubits, ancilla2_qubits):
    """Prepare Arthurs-Kelly ancillas."""
    yield from qft(ancilla2_qubits, inverse=True)
    yield from CZ(ancilla1_qubits, ancilla2_qubits)


def arthurs_kelly_coupling(system_qubits, ancilla1_qubits, ancilla2_qubits):
    """Qudit Arthurs-Kelly coupling."""
    yield from CX(system_qubits, ancilla1_qubits, inverse=True)
    yield from qft(system_qubits, inverse=True)
    yield from CX(system_qubits, ancilla2_qubits, inverse=True)
    yield from qft(system_qubits)


def arthurs_kelly(
    system_qubits,
    ancilla1_qubits,
    ancilla2_qubits,
    prepare_fiducial=None,
    prepare_ancillas=None,
    measure=True,
):
    """Qudit Arthurs-Kelly on n qubits with two n qubit ancillas."""
    if prepare_fiducial is not None:
        yield from prepare_fiducial(ancilla1_qubits, conjugate=True)
        yield from prepare_fiducial(ancilla2_qubits)
        yield from ready_arthurs_kelly_ancillas(ancilla1_qubits, ancilla2_qubits)
    if prepare_ancillas is not None:
        yield from prepare_ancillas(ancilla1_qubits, ancilla2_qubits)
    yield from arthurs_kelly_coupling(system_qubits, ancilla1_qubits, ancilla2_qubits)
    if measure:
        yield measure_register(ancilla1_qubits, "a1")
        yield measure_register(ancilla2_qubits, "a2")


####################################################################################


def simple_wh_povm(system_qubits, ancilla_qubits, prepare_fiducial=None, measure=True):
    """Simple WH-POVM on n qubits with n qubit ancilla."""
    if prepare_fiducial is not None:
        yield prepare_fiducial(ancilla_qubits, conjugate=True)
    yield CX(ancilla_qubits, system_qubits, inverse=True)
    yield qft(ancilla_qubits, inverse=True)
    if measure:
        yield measure_register(system_qubits, "s")
        yield measure_register(ancilla_qubits, "a")


def msimple_wh_povm(system_qubits, ancilla_qubits, prepare_fiducial=None, measure=True):
    """Measurement-based simple WH-POVM with mid-circuit feed-forward."""
    n = len(system_qubits)

    if prepare_fiducial is not None:
        yield prepare_fiducial(ancilla_qubits, conjugate=True)

    if not measure:
        yield from simple_wh_povm(
            system_qubits, ancilla_qubits, prepare_fiducial=None, measure=False
        )
        return

    system_keys = {q: f"s_{i}" for i, q in enumerate(system_qubits)}
    ancilla_keys = {q: f"a_{i}" for i, q in enumerate(ancilla_qubits)}

    yield from qft(system_qubits)
    yield from CZ(ancilla_qubits, system_qubits, inverse=True)
    yield from mqft(system_qubits, inverse=True, key_fn=system_keys.__getitem__)
    yield from mqft(ancilla_qubits, inverse=True, key_fn=ancilla_keys.__getitem__)


####################################################################################


def CRy(theta):
    """Controlled y-rotation."""

    def __CRy__(control, target):
        yield cirq.Ry(rads=theta / 2)(target)
        yield cirq.CNOT(control, target)
        yield cirq.Ry(rads=-theta / 2)(target)
        yield cirq.CNOT(control, target)

    return __CRy__


def d4_sic_monomial_fiducial(qubits):
    """Prepare an almost flat d=4 monomial SIC fiducial."""
    theta1 = 2 * np.arccos(np.sqrt((5 + np.sqrt(5)) / 10))
    theta2 = 2 * np.arccos(np.sqrt(1 + np.sqrt(5)) / 2)
    theta3 = np.pi / 2

    yield cirq.Ry(rads=theta1)(qubits[0])
    yield cirq.X(qubits[0])
    yield CRy(theta2)(qubits[0], qubits[1])
    yield cirq.X(qubits[0])
    yield CRy(theta3)(qubits[0], qubits[1])


def d4_monomial_rephasing(qubits):
    """Rephase the d=4 monomial basis."""
    yield cirq.ZPowGate(exponent=1)(qubits[1])
    yield cirq.CNOT(qubits[0], qubits[1])
    yield cirq.ZPowGate(exponent=3 / 4)(qubits[1])
    yield cirq.CNOT(qubits[0], qubits[1])
    yield cirq.ZPowGate(exponent=-1 / 2)(qubits[0])


def d4_sic_fiducial(qubits, conjugate=False):
    """Prepare a d=4 SIC fiducial (or its conjugate)."""
    yield from d4_sic_monomial_fiducial(qubits)
    yield from d4_monomial_rephasing(qubits) if not conjugate else cirq.inverse(
        list(d4_monomial_rephasing(qubits))
    )
    yield cirq.H(qubits[0])


####################################################################################


def __ansatz_circuit__(q, params, conjugate=False):
    """Implements the Grey code ansatz as a function of params."""
    n = len(q)
    targeting_data = [grey_data(i) for i in range(n)]
    sign = -1 if conjugate else 1
    thetas, phis, phase = ansatz_params_to_angles(n, params, sign=sign)

    yield cirq.Rz(rads=-2 * sign * phase)(q[0])
    yield cirq.Ry(rads=thetas[0][0])(q[0])
    yield cirq.Rz(rads=phis[0][0])(q[0])
    for i in range(1, len(thetas)):
        current_q = q[: i + 1]
        M, targets = targeting_data[i]
        for j, theta in enumerate(M @ thetas[i]):
            yield cirq.Ry(rads=theta)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])
        for j, phi in enumerate(M @ phis[i]):
            yield cirq.Rz(rads=phi)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])


def ansatz_circuit(ket):
    """Return a gate generator preparing a logical ket on a qubit register.

    Non-power-of-two vectors are zero-padded to the supplied register.
    ``conjugate=True`` prepares the componentwise complex conjugate. The
    Gray-code decomposition uses only one-qubit rotations and CNOTs.
    """
    from .measurements import _validated_state, embed_state

    ket = _validated_state(ket)
    if ket.ndim != 1:
        raise ValueError("ansatz_circuit requires a ket")

    def prepare(q, conjugate=False):
        padded = embed_state(ket, len(q))
        params = ansatz_angles_to_params(*ket_to_ansatz_angles(padded))
        yield from __ansatz_circuit__(q, params, conjugate=conjugate)

    return prepare


def _embedded_matrix(U, qubits):
    """Validate and extend a unitary by identity on unused basis states."""
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("U must be a nonempty square matrix")
    if not np.all(np.isfinite(U)) or not np.allclose(
        U.conj().T @ U, np.eye(len(U)), atol=1e-9, rtol=0
    ):
        raise ValueError("U must be unitary")
    size = 2 ** len(qubits)
    if len(U) > size:
        raise ValueError("logical dimension does not fit the qubit register")
    return block_diag(U, np.eye(size - len(U)))


def Z_d(d, qubits, aux=(), k=1):
    """Apply the dimension-d clock, identity on padding; ``aux`` is unused."""
    d = _dimension(d)
    phases = np.exp(2j * np.pi * k * np.arange(d) / d)
    yield from embed_gate(np.diag(phases), qubits)


def X_d(d, qubits, aux=(), k=1):
    """Apply ``|j> -> |j+k mod d>`` on the logical subspace.

    Padding states are fixed. ``aux`` is retained for source compatibility
    and may be an empty tuple; it is never touched.
    """
    d = _dimension(d)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("shift exponent k must be an integer")
    yield from embed_gate(np.roll(np.eye(d, dtype=complex), k % d, axis=0), qubits)


def displace_d(d, qubits, aux, a1, a2):
    """Qudit displacement operator on n qudits with indices (a1, a2)."""
    yield from Z_d(d, qubits, aux, a2)
    yield from X_d(d, qubits, aux, a1)


def wh_state_d(d, qubits, aux, prepare_fiducial, a1, a2):
    """Prepare the WH state D(a1,a2)|fiducial> on n qubits."""
    yield from prepare_fiducial(qubits)
    yield from displace_d(d, qubits, aux, a1, a2)


####################################################################################


def _controlled_qudit_matrix(d, control_qubits, target_qubits, *, shift, inverse):
    """Controlled logical operation; fix any input with a padded register."""
    d = _dimension(d)
    nc, nt = 2 ** len(control_qubits), 2 ** len(target_qubits)
    if d > min(nc, nt):
        raise ValueError("logical dimension does not fit both registers")
    size = nc * nt
    matrix = np.eye(size, dtype=complex)
    sign = -1 if inverse else 1
    for control in range(d):
        for target in range(d):
            column = control * nt + target
            if shift:
                matrix[column, column] = 0
                matrix[control * nt + (target + sign * control) % d, column] = 1
            else:
                matrix[column, column] = np.exp(
                    sign * 2j * np.pi * control * target / d
                )
    return matrix


def CZ_d(d, control_qubits, target_qubits, aux=(), inverse=False):
    """Apply logical controlled clock; padded controls/targets are fixed."""
    matrix = _controlled_qudit_matrix(
        d, control_qubits, target_qubits, shift=False, inverse=inverse
    )
    yield cirq.MatrixGate(matrix, name=f"CZ({d})").on(
        *(tuple(control_qubits) + tuple(target_qubits))
    )


def CX_d(d, control_qubits, target_qubits, aux=(), inverse=False):
    """Apply ``|c,t> -> |c,t+c mod d>`` within the logical subspace.

    The inverse subtracts ``c``. Inputs with either register at least ``d``
    are unchanged, and the legacy ``aux`` register is untouched.
    """
    matrix = _controlled_qudit_matrix(
        d, control_qubits, target_qubits, shift=True, inverse=inverse
    )
    yield cirq.MatrixGate(matrix, name=f"CX({d})").on(
        *(tuple(control_qubits) + tuple(target_qubits))
    )


def qft_d(d, qubits, inverse=False):
    """Apply the positive-exponent dimension-d DFT, identity on padding."""
    d = _dimension(d)
    F = np.exp(2j * np.pi * np.outer(np.arange(d), np.arange(d)) / d) / np.sqrt(d)
    yield from embed_gate(F, qubits, inverse=inverse)


def embed_gate(U, qubits, inverse=False):
    """Apply ``U`` to the first d basis states and identity to padding.

    Returns a MatrixGate operation without choosing a hardware gate set.
    Compile it using the target device's Cirq transformer when needed.
    """
    matrix = _embedded_matrix(U, qubits)
    if inverse:
        matrix = matrix.conj().T
    yield cirq.MatrixGate(matrix).on(*qubits)


def ready_arthurs_kelly_ancillas_d(d, ancilla1_qubits, ancilla2_qubits, aux):
    """Prepare Arthurs-Kelly ancillas."""
    yield from qft_d(d, ancilla2_qubits, inverse=True)
    yield from CZ_d(d, ancilla1_qubits, ancilla2_qubits, aux)


def arthurs_kelly_coupling_d(d, system_qubits, ancilla1_qubits, ancilla2_qubits, aux):
    """Qudit Arthurs-Kelly coupling."""
    yield from CX_d(d, system_qubits, ancilla1_qubits, aux, inverse=True)
    yield from qft_d(d, system_qubits, inverse=True)
    yield from CX_d(d, system_qubits, ancilla2_qubits, aux, inverse=True)
    yield from qft_d(d, system_qubits)


def arthurs_kelly_d(
    d,
    system_qubits,
    ancilla1_qubits,
    ancilla2_qubits,
    aux,
    prepare_fiducial=None,
    prepare_ancillas=None,
    measure=True,
):
    """Qudit Arthurs-Kelly on n qubits with two n qubit ancillas."""
    if prepare_fiducial is not None:
        yield from prepare_fiducial(ancilla1_qubits, conjugate=True)
        yield from prepare_fiducial(ancilla2_qubits)
        yield from ready_arthurs_kelly_ancillas_d(
            d, ancilla1_qubits, ancilla2_qubits, aux
        )
    if prepare_ancillas is not None:
        yield from prepare_ancillas(d, ancilla1_qubits, ancilla2_qubits, aux)
    yield from arthurs_kelly_coupling_d(
        d, system_qubits, ancilla1_qubits, ancilla2_qubits, aux
    )
    if measure:
        yield measure_register(ancilla1_qubits, "a1")
        yield measure_register(ancilla2_qubits, "a2")


####################################################################################


def simple_wh_povm_d(
    d, system_qubits, ancilla_qubits, aux, prepare_fiducial=None, measure=True
):
    """Legacy WH-POVM on separately embedded system and ancilla registers.

    The outcome ``(a,b)`` occupies binary integer ``a*2**n+b`` here, so
    padding is interleaved; ``aux`` is retained but unused. Prefer
    ``build_measurement_circuit`` for contiguous ``k=a*d+b`` outcomes.
    """
    if prepare_fiducial is not None:
        yield prepare_fiducial(ancilla_qubits, conjugate=True)
    yield from CX_d(d, ancilla_qubits, system_qubits, aux, inverse=True)
    yield from qft_d(d, ancilla_qubits, inverse=True)
    if measure:
        yield measure_register(system_qubits, "s")
        yield measure_register(ancilla_qubits, "a")


####################################################################################


def get_triple_product_qubits(d):
    """Convenient control and three embedded logical registers in dimension d."""
    d = _dimension(d)
    n = max(1, (d - 1).bit_length())
    grid = cirq.GridQubit.rect(4, n, top=4, left=2)
    rows = [grid[i * n : (i + 1) * n] for i in range(4)]
    control_qubit = rows[3][0]
    return [control_qubit] + rows[0] + rows[1] + rows[2]


def _two_qubit_decomposition(op):
    """Decompose an operation to one- and two-qubit gates when possible."""
    return cirq.decompose(
        op,
        keep=lambda inner: cirq.is_measurement(inner) or len(inner.qubits) <= 2,
        on_stuck_raise=ValueError,
    )


def controlled_register_swap(control_qubit, register1, register2):
    """Controlled swap between two ``n``-qubit registers, pairwise by qubit."""
    if len(register1) != len(register2):
        raise ValueError("Controlled register swap requires equal-length registers.")
    for qubit1, qubit2 in zip(register1, register2):
        yield from _two_qubit_decomposition(cirq.CSWAP(control_qubit, qubit1, qubit2))


def controlled_cyclic_permutation(control_qubit, register1, register2, register3):
    """Controlled three-cycle ``(1 2 3)`` on equal-length qubit registers."""
    if not (len(register1) == len(register2) == len(register3)):
        raise ValueError(
            "Controlled cyclic permutation requires equal-length registers."
        )

    # Two register swaps implement the left cycle:
    # (register1, register2, register3) -> (register2, register3, register1).
    yield from controlled_register_swap(control_qubit, register1, register3)
    yield from controlled_register_swap(control_qubit, register1, register2)


def triple_product_measurement_circuit(
    control_qubit,
    register1,
    register2,
    register3,
    prepare_state1,
    prepare_state2,
    prepare_state3,
    *,
    measure=True,
):
    """Cycle-test circuit for the real part of ``tr(sigma1 sigma2 sigma3)``."""
    if not (len(register1) == len(register2) == len(register3)):
        raise ValueError("Triple-product circuit requires equal-length registers.")

    circuit = cirq.Circuit(
        prepare_state1(register1),
        prepare_state2(register2),
        prepare_state3(register3),
        cirq.H(control_qubit),
        controlled_cyclic_permutation(control_qubit, register1, register2, register3),
        cirq.H(control_qubit),
    )
    if measure:
        circuit.append(cirq.measure(control_qubit, key="result"))
    return circuit


def real_triple_product_from_probabilities(probabilities):
    """Recover ``Re(tr(sigma1 sigma2 sigma3))`` from cycle-test probabilities."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 0 or probabilities.shape[-1] != 2:
        raise ValueError("Expected binary probabilities with shape (..., 2).")
    if (
        not np.all(np.isfinite(probabilities))
        or np.iscomplexobj(probabilities)
        or np.any(probabilities < -1e-9)
        or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "Binary probabilities must be finite, real, nonnegative and normalized."
        )
    return probabilities[..., 0] - probabilities[..., 1]
