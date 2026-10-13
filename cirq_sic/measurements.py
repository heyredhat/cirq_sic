"""Rank-one POVMs and their exact embeddings on qubit hardware.

A frame has shape ``(d, m)`` and **weighted** columns ``f_k`` such that
``sum_k |f_k><f_k| = I``. The outcome ``k`` therefore has effect
``E_k = |f_k><f_k|``. For WH measurements, ``k = a*d+b`` and
``f_k = X**a Z**b phi / sqrt(d)``.

The Naimark construction uses ``ceil(log2(m))`` qubits (at least one).
Logical input basis state ``j`` occupies the first ``d`` states of the
system register; appended ancillas start at zero. Output integers below
``m`` are measurement outcomes, and all remaining integers are padding.
Padding is observable leakage, never silently postselected away.

This is an exact, dense reference construction. Its unitary uses O(m²)
memory and its completion O(m³) time; it is not a scalable gate synthesis
algorithm. Hardware users should compile the resulting MatrixGate to
their device's gate set and inspect the resulting depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import cirq
import numpy as np
from scipy.linalg import qr

from .wh import _dimension, wh_frame


def _qubit_count(dimension):
    """Qubit count for a positive integer dimension, including dimension one."""
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(
        dimension, (int, np.integer)
    ):
        raise ValueError("dimension must be a positive integer")
    if dimension < 1:
        raise ValueError("dimension must be a positive integer")
    return max(1, (int(dimension) - 1).bit_length())


def _validated_state(state, dimension=None, *, atol=1e-9):
    """Copy a finite, normalized ket or positive trace-one density matrix."""
    state = np.array(state, dtype=complex, copy=True)
    if state.ndim not in (1, 2) or state.shape[0] == 0:
        raise ValueError("state must be a nonempty ket or square density matrix")
    if dimension is not None and state.shape[0] != dimension:
        raise ValueError(f"state dimension must be {dimension}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain finite values")
    if state.ndim == 1:
        if not np.isclose(np.vdot(state, state), 1, atol=atol, rtol=0):
            raise ValueError("ket must have unit norm")
    else:
        if state.shape[0] != state.shape[1]:
            raise ValueError("density matrix must be square")
        if not np.allclose(state, state.conj().T, atol=atol, rtol=0):
            raise ValueError("density matrix must be Hermitian")
        if not np.isclose(np.trace(state), 1, atol=atol, rtol=0):
            raise ValueError("density matrix must have trace one")
        if np.linalg.eigvalsh(state).min() < -atol:
            raise ValueError("density matrix must be positive semidefinite")
    return state


def embed_state(state, num_qubits=None):
    """Zero-pad a logical ket or density matrix into a qubit register.

    The first ``d`` computational basis states encode the logical system,
    in big-endian qubit order. Inputs must already be normalized. No
    renormalization or projection is performed.
    """
    state = _validated_state(state)
    required = _qubit_count(state.shape[0])
    if num_qubits is None:
        num_qubits = required
    if (
        isinstance(num_qubits, (bool, np.bool_))
        or not isinstance(num_qubits, (int, np.integer))
        or num_qubits < required
    ):
        raise ValueError(f"num_qubits must be an integer at least {required}")
    size = 2 ** int(num_qubits)
    if state.ndim == 1:
        padded = np.zeros(size, dtype=complex)
        padded[: len(state)] = state
    else:
        padded = np.zeros((size, size), dtype=complex)
        padded[: len(state), : len(state)] = state
    return padded


def _complete_isometry(isometry, input_indices=None):
    """Complete orthonormal columns without changing their phases or order."""
    rows, columns = isometry.shape
    basis, _ = qr(isometry, mode="full")
    if input_indices is None:
        input_indices = np.arange(columns)
    input_indices = np.asarray(input_indices)
    remaining = np.setdiff1d(np.arange(rows), input_indices)
    unitary = np.empty((rows, rows), dtype=complex)
    unitary[:, input_indices] = isometry
    unitary[:, remaining] = basis[:, columns:]
    return unitary


def prepare_state(state, system_qubits, *, environment_qubits=()):
    """Return a unitary preparation operation acting on initially zero qubits.

    A ket is embedded in ``system_qubits``. A density matrix is purified as
    ``sum_i sqrt(lambda_i) |v_i>_system |i>_environment`` and requires an
    equally sized environment register, which must not be measured or
    reused as measurement ancillas. Tracing it out prepares the requested
    density matrix. The preparation preserves the ket's global phase.
    """
    state = _validated_state(state)
    system_qubits = tuple(system_qubits)
    environment_qubits = tuple(environment_qubits)
    _validate_qubits(system_qubits + environment_qubits)
    if state.ndim == 1:
        if environment_qubits:
            raise ValueError("a pure ket does not require environment_qubits")
        vector = embed_state(state, len(system_qubits))
    else:
        if len(environment_qubits) != len(system_qubits):
            raise ValueError(
                "mixed-state preparation requires one environment qubit per system qubit"
            )
        size = 2 ** len(system_qubits)
        embed_state(state, len(system_qubits))  # Validate capacity before allocating.
        eigenvalues, eigenvectors = np.linalg.eigh(state)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues /= eigenvalues.sum()
        purification = np.zeros((size, size), dtype=complex)
        purification[: len(state), : len(state)] = eigenvectors * np.sqrt(eigenvalues)
        vector = purification.reshape(-1)
    unitary = _complete_isometry(vector[:, None])
    return cirq.MatrixGate(unitary, name="Prepare").on(
        *(system_qubits + environment_qubits)
    )


def _validate_qubits(qubits):
    """Check distinct, binary Cirq qids in the caller's explicit order."""
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubit registers must be disjoint and contain distinct qubits")
    if any(not isinstance(qubit, cirq.Qid) or qubit.dimension != 2 for qubit in qubits):
        raise ValueError("all register entries must be binary Cirq qubits")


class Measurement:
    """A validated rank-one POVM represented by its weighted frame.

    Parameters
    ----------
    frame : array_like, shape (d, m)
        Nonzero columns ``f_k`` satisfying ``frame @ frame.conj().T = I``.
        Ordinary normalized WH orbit vectors must first be divided by
        ``sqrt(d)``; :func:`wh_measurement` performs that scaling.
    name : str
        Human-readable name used in circuit diagrams and saved metadata.
    atol : float
        Absolute tolerance for completeness checks.

    Arrays are copied and exposed read-only. Outcome order is exactly the
    order of the frame columns. General rank-one POVMs, including an
    orthonormal basis measurement, use the same circuit implementation.
    """

    def __init__(self, frame, name="POVM", *, atol=1e-9):
        if not np.isfinite(atol) or atol <= 0:
            raise ValueError("atol must be positive and finite")
        frame = np.array(frame, dtype=complex, copy=True)
        if frame.ndim != 2 or min(frame.shape) < 1:
            raise ValueError("frame must have nonempty shape (dimension, outcomes)")
        if not np.all(np.isfinite(frame)):
            raise ValueError("frame must contain finite values")
        if np.any(np.linalg.norm(frame, axis=0) == 0):
            raise ValueError("frame columns must be nonzero")
        if not np.allclose(
            frame @ frame.conj().T, np.eye(frame.shape[0]), atol=atol, rtol=0
        ):
            raise ValueError(
                "frame is not a complete POVM: frame @ frame† must equal identity"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        frame.setflags(write=False)
        self._frame = frame
        self.name = name

    @property
    def frame(self):
        """Read-only ``(dimension, num_outcomes)`` array of weighted kets."""
        return self._frame

    @property
    def dimension(self):
        """Logical Hilbert-space dimension."""
        return self.frame.shape[0]

    @property
    def num_outcomes(self):
        """Number of physical POVM effects, excluding padded bit strings."""
        return self.frame.shape[1]

    @property
    def num_system_qubits(self):
        """Minimum qubits encoding the logical input system."""
        return _qubit_count(self.dimension)

    @property
    def num_qubits(self):
        """Total measured qubits, including the system and zero ancillas."""
        return _qubit_count(self.num_outcomes)

    @property
    def num_ancilla_qubits(self):
        """Zero ancillas appended to the system for the measurement."""
        return self.num_qubits - self.num_system_qubits

    @cached_property
    def effects(self):
        """Read-only effects, shape ``(num_outcomes, dimension, dimension)``."""
        effects = np.einsum("ik,jk->kij", self.frame, self.frame.conj())
        effects.setflags(write=False)
        return effects

    @cached_property
    def states(self):
        """Read-only unit-norm frame columns, shape ``(dimension, outcomes)``."""
        states = self.frame / np.linalg.norm(self.frame, axis=0)
        states.setflags(write=False)
        return states

    def probabilities(self, state):
        """Born probabilities for a normalized logical ket or density matrix."""
        state = _validated_state(state, self.dimension)
        if state.ndim == 1:
            return np.abs(self.frame.conj().T @ state) ** 2
        return np.einsum("ik,ij,jk->k", self.frame.conj(), state, self.frame).real

    def isometry(self):
        """Return the padded Naimark isometry, shape ``(2**num_qubits, d)``."""
        isometry = np.zeros((2**self.num_qubits, self.dimension), dtype=complex)
        isometry[: self.num_outcomes] = self.frame.conj().T
        return isometry

    @cached_property
    def _unitary(self):
        """Cache the dense dilation only when a circuit actually needs it."""
        input_indices = np.arange(self.dimension) * 2**self.num_ancilla_qubits
        unitary = _complete_isometry(self.isometry(), input_indices)
        unitary.setflags(write=False)
        return unitary

    def unitary(self):
        """Return the read-only dilation with system-first, ancilla-last input.

        Column ``j * 2**num_ancilla_qubits`` equals isometry column ``j``.
        Other input columns are an arbitrary orthonormal completion; they
        are outside the valid logical input subspace.
        """
        return self._unitary

    def is_sic(self, *, atol=1e-7):
        """Whether this frame has d² equal-weight equiangular SIC effects.

        This generic check allocates a d²-by-d² overlap matrix and is
        intended for small dimensions. The WH-specific ``sic_povm`` factory
        uses covariance to perform the same validation more economically.
        """
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative")
        d = self.dimension
        if self.num_outcomes != d * d:
            return False
        if not np.allclose(
            np.sum(np.abs(self.frame) ** 2, axis=0), 1 / d, atol=atol, rtol=0
        ):
            return False
        overlaps = np.abs(self.states.conj().T @ self.states) ** 2
        return bool(
            np.allclose(overlaps, (d * np.eye(d * d) + 1) / (d + 1), atol=atol, rtol=0)
        )

    def circuit(self, state=None, **kwargs):
        """Build a :class:`MeasurementCircuit`; see :func:`build_measurement_circuit`."""
        return build_measurement_circuit(self, state, **kwargs)

    def __repr__(self):
        return f"Measurement(name={self.name!r}, dimension={self.dimension}, num_outcomes={self.num_outcomes})"


def wh_measurement(fiducial):
    """Construct a WH POVM with outcomes ``k=a*d+b`` for ``D=X**a Z**b``.

    The fiducial must be a unit ket. Any such ket gives a complete POVM;
    informational completeness and the SIC overlap conditions are separate
    properties and are not assumed.
    """
    fiducial = _validated_state(fiducial)
    if fiducial.ndim != 1:
        raise ValueError("fiducial must be a ket")
    return Measurement(wh_frame(fiducial) / np.sqrt(len(fiducial)), name="WH-POVM")


def sic_povm(d=None, *, fiducial=None, atol=1e-7):
    """Construct and validate a WH-covariant SIC from data or a supplied ket.

    ``d`` loads a bundled fiducial; supplying ``fiducial`` supports any
    dimension for which a SIC fiducial is known. This function does not
    claim to construct unknown SICs in every dimension. A supplied ``d``
    must match the fiducial length. Pairwise squared overlaps are checked
    against ``(d*delta_ij+1)/(d+1)`` before returning the measurement.
    Bundled fiducials through dimension 32 are numerically refined and
    cached if needed: some original text files have only modest accuracy.
    Higher dimensions and user-supplied fiducials are never automatically
    optimized; use ``refine_sic_fiducial`` explicitly when required.
    """
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be positive and finite")
    if d is not None:
        d = _dimension(d)
    if fiducial is None:
        if d is None:
            raise ValueError("provide d or fiducial")
        from .sics import _measurement_fiducial

        fiducial = np.ones(1, dtype=complex) if d == 1 else _measurement_fiducial(d)
    else:
        fiducial = _validated_state(fiducial)
        if fiducial.ndim != 1:
            raise ValueError("fiducial must be a ket")
        if d is not None and d != len(fiducial):
            raise ValueError("d does not match fiducial dimension")
    measurement = wh_measurement(fiducial)
    d = measurement.dimension
    # WH covariance makes all pairwise overlaps translates of this row;
    # checking it avoids allocating a d² by d² Gram matrix.
    overlaps = np.abs(measurement.states[:, 0].conj() @ measurement.states) ** 2
    expected = np.full(d * d, 1 / (d + 1))
    expected[0] = 1
    if not np.allclose(overlaps, expected, atol=atol, rtol=0):
        error = np.max(np.abs(overlaps - expected))
        raise ValueError(
            f"fiducial does not generate a SIC within tolerance (overlap error {error:.3g}); "
            "use refine_sic_fiducial or choose an explicit approximation tolerance"
        )
    measurement.name = "SIC-POVM"
    return measurement


@dataclass(frozen=True)
class MeasurementCircuit:
    """Circuit plus the qubit and outcome order needed to interpret its data.

    ``measured_qubits`` are big-endian; their integer ``k`` is the frame
    column index. Integers at least ``measurement.num_outcomes`` are padding
    and must remain in saved counts. Environment qubits are traced out.
    """

    circuit: cirq.Circuit
    measurement: Measurement
    system_qubits: tuple
    ancilla_qubits: tuple
    environment_qubits: tuple
    measurement_key: str

    @property
    def measured_qubits(self):
        """System followed by measurement ancillas, in output bit order."""
        return self.system_qubits + self.ancilla_qubits

    @property
    def qubit_order(self):
        """Complete simulation order, with unmeasured environment last."""
        return self.measured_qubits + self.environment_qubits

    @property
    def outcome_indices(self):
        """Valid logical output integers; all other integers are padding."""
        return tuple(range(self.measurement.num_outcomes))


def build_measurement_circuit(
    measurement,
    state=None,
    *,
    system_qubits: Sequence[cirq.Qid] | None = None,
    ancilla_qubits: Sequence[cirq.Qid] | None = None,
    environment_qubits: Sequence[cirq.Qid] | None = None,
    key="outcome",
    measure=True,
):
    """Build preparation, Naimark dilation and optional terminal measurement.

    With ``state=None``, the caller supplies a state on the system register;
    measurement ancillas must begin in zero. A ket adds unitary preparation.
    A density matrix adds purification and an unmeasured environment with
    one qubit per system qubit. Default registers use distinct LineQubits.
    Custom system and ancilla registers must have the minimum sizes exposed
    by ``measurement``. Use the returned explicit order in every simulation.
    """
    if not isinstance(measurement, Measurement):
        raise TypeError("measurement must be a Measurement")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a nonempty string")
    if state is not None:
        state = _validated_state(state, measurement.dimension)
    ns = measurement.num_system_qubits
    na = measurement.num_ancilla_qubits
    ne = ns if state is not None and state.ndim == 2 else 0
    registers = [system_qubits, ancilla_qubits, environment_qubits]
    lengths = [ns, na, ne]
    used = {
        qubit for register in registers if register is not None for qubit in register
    }
    next_index = 0
    for index, (register, length) in enumerate(zip(registers, lengths)):
        if register is None:
            register = []
            while len(register) < length:
                qubit = cirq.LineQubit(next_index)
                next_index += 1
                if qubit not in used:
                    register.append(qubit)
                    used.add(qubit)
        register = tuple(register)
        if len(register) != length:
            raise ValueError(f"register {index} requires {length} qubits")
        registers[index] = register
    system, ancillas, environment = registers
    _validate_qubits(system + ancillas + environment)
    circuit = cirq.Circuit()
    if state is not None:
        circuit.append(prepare_state(state, system, environment_qubits=environment))
    measured = system + ancillas
    circuit.append(
        cirq.MatrixGate(measurement.unitary(), name=measurement.name).on(*measured)
    )
    if measure:
        circuit.append(cirq.measure(*measured, key=key))
    return MeasurementCircuit(circuit, measurement, system, ancillas, environment, key)


def bell_overlap_circuit(state, *, key="outcome", measure=True):
    """Measure cyclic-WH displaced overlaps of a state with a second copy.

    Prepare ``rho`` and its computational-basis conjugate ``rho.conj()``;
    a generalized Bell measurement then yields
    ``s[a*d+b] = Tr(rho D[a,b] rho D[a,b]†)/d``. In particular,
    ``d*s[0] = Tr(rho**2)``. The same unitary construction works for pure
    kets and mixed states (each mixed copy gets its own purification).
    Preparing the conjugate requires a known preparation description;
    this is not a physical conjugation channel on an unknown input.

    Two separate ``ceil(log2(d))``-qubit registers are encoded, so this
    specialized layout may use one more measured qubit than a compact
    encoding of dimension ``d²``. Output integers ``0..d²-1`` are contiguous
    WH labels; every other measured integer is padding. The returned
    Measurement represents the Bell basis on the abstract joint dimension
    ``d²``; use the returned circuit's explicit qubit layout for execution.
    """
    state = _validated_state(state)
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a nonempty string")
    d = len(state)
    n = _qubit_count(d)
    size = 2**n
    measured = tuple(cirq.LineQubit.range(2 * n))
    first, second = measured[:n], measured[n:]
    environment = tuple(cirq.LineQubit.range(2 * n, 4 * n)) if state.ndim == 2 else ()
    first_environment, second_environment = environment[:n], environment[n:]
    circuit = cirq.Circuit(
        prepare_state(state, first, environment_qubits=first_environment),
        prepare_state(state.conj(), second, environment_qubits=second_environment),
    )
    bell = np.zeros((d * d, d * d), dtype=complex)
    y = np.arange(d)
    for a in range(d):
        for b in range(d):
            bell[a * d + b, ((y + a) % d) * d + y] = np.exp(
                -2j * np.pi * b * y / d
            ) / np.sqrt(d)
    joint_measurement = Measurement(bell.conj().T, name="WH-Bell")
    logical_columns = (np.arange(d)[:, None] * size + np.arange(d)).reshape(-1)
    unused_columns = np.setdiff1d(np.arange(size * size), logical_columns)
    unitary = np.zeros((size * size, size * size), dtype=complex)
    unitary[: d * d, logical_columns] = bell
    unitary[np.arange(d * d, size * size), unused_columns] = 1
    circuit.append(cirq.MatrixGate(unitary, name="WH-Bell").on(*measured))
    if measure:
        circuit.append(cirq.measure(*measured, key=key))
    return MeasurementCircuit(
        circuit, joint_measurement, measured, (), environment, key
    )


__all__ = [
    "Measurement",
    "MeasurementCircuit",
    "build_measurement_circuit",
    "embed_state",
    "prepare_state",
    "sic_povm",
    "wh_measurement",
    "bell_overlap_circuit",
]
