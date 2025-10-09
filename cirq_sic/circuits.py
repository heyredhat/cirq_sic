import cirq 
import numpy as np

from .ansatz import *

def qudit_basis_state(qubits, m):
    """Prepares the qudit basis state |m> on the qubits q."""
    n = len(qubits)
    bitstr = bin(m)[2:].zfill(n)
    for i, b in enumerate(bitstr):
        if b == '1':
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
            yield cirq.CZPowGate(exponent=2**(i-j))(qubits[j], qubits[i])
    for i in range(n // 2):
        yield cirq.SWAP(qubits[i], qubits[n - 1 - i])

def Z(qubits, k=1):
    """Qudit shift on n qubits."""
    for j, qubit in enumerate(qubits):
        angle = k * np.pi / (2**j)
        if angle != 0:
            yield cirq.ZPowGate(exponent=angle/np.pi)(qubits[j])

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
            yield cirq.CZPowGate(exponent=angle/np.pi)(control_qubit, target_qubit)

def CZ(control_qubits, target_qubits, inverse=False):
    """Qudit controlled clock on n control qubit and n target qubits."""
    if inverse:
        yield from cirq.inverse(list(CZ(control_qubits, target_qubits)))
        return 
    n = len(control_qubits)
    for j in range(n):
        yield from QCZ(control_qubits[n-j-1], target_qubits, 2**j)

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

def arthurs_kelly(system_qubits, ancilla1_qubits, ancilla2_qubits, prepare_fiducial=None, prepare_ancillas=None, measure=True):
    """Qudit Arthurs-Kelly on n qubits with two n qubit ancillas."""
    if prepare_fiducial is not None:
        yield from prepare_fiducial(ancilla1_qubits, conjugate=True)
        yield from prepare_fiducial(ancilla2_qubits)
        yield from ready_arthurs_kelly_ancillas(ancilla1_qubits, ancilla2_qubits)
    if prepare_ancillas is not None:
        yield from prepare_ancillas(ancilla1_qubits, ancilla2_qubits)
    yield from arthurs_kelly_coupling(system_qubits, ancilla1_qubits, ancilla2_qubits)
    if measure:
        yield cirq.measure(*[ancilla1_qubits+ancilla2_qubits], key="result")

####################################################################################

def simple_wh_povm(system_qubits, ancilla_qubits, prepare_fiducial=None, measure=True):
    """Simple WH-POVM on n qubits with n qubit ancilla."""
    if prepare_fiducial is not None:
        yield prepare_fiducial(ancilla_qubits, conjugate=True)
    yield CX(ancilla_qubits, system_qubits, inverse=True)
    yield qft(ancilla_qubits, inverse=True)
    if measure:
        yield cirq.measure(*(system_qubits+ancilla_qubits), key="result")

####################################################################################

def CRy(theta):
	"""Controlled y-rotation."""
	def __CRy__(control, target):
		yield cirq.Ry(rads=theta/2)(target)
		yield cirq.CNOT(control, target)
		yield cirq.Ry(rads=-theta/2)(target)
		yield cirq.CNOT(control, target)
	return __CRy__

def d4_sic_monomial_fiducial(qubits):
    """Prepare an almost flat d=4 monomial SIC fiducial."""
    theta1 = 2*np.arccos(np.sqrt((5+np.sqrt(5))/10))
    theta2 = 2*np.arccos(np.sqrt(1+np.sqrt(5))/2)
    theta3 = np.pi/2

    yield cirq.Ry(rads=theta1)(qubits[0])
    yield cirq.X(qubits[0])
    yield CRy(theta2)(qubits[0], qubits[1])
    yield cirq.X(qubits[0])
    yield CRy(theta3)(qubits[0], qubits[1])

def d4_monomial_rephasing(qubits):
    """Rephase the d=4 monomial basis."""
    yield cirq.ZPowGate(exponent=1)(qubits[1])
    yield cirq.CNOT(qubits[0], qubits[1])
    yield cirq.ZPowGate(exponent=3/4)(qubits[1])
    yield cirq.CNOT(qubits[0], qubits[1])
    yield cirq.ZPowGate(exponent=-1/2)(qubits[0])

def d4_sic_fiducial(qubits, conjugate=False):
	"""Prepare a d=4 SIC fiducial (or its conjugate)."""
	yield from d4_sic_monomial_fiducial(qubits)
	yield from d4_monomial_rephasing(qubits) if not conjugate else \
		  cirq.inverse(list(d4_monomial_rephasing(qubits)))
	yield cirq.H(qubits[0])
     
####################################################################################

def __ansatz_circuit__(q, params, conjugate=False):
    n = len(q)
    targeting_data = [grey_data(i) for i in range(n)]
    sign = -1 if conjugate else 1
    thetas, phis, phase = ansatz_params_to_angles(n, params, sign=sign)
    
    yield cirq.Rz(rads=-2*sign*phase)(q[0])
    yield cirq.Ry(rads=thetas[0][0])(q[0])
    yield cirq.Rz(rads=phis[0][0])(q[0])
    for i in range(1, len(thetas)):
        current_q = q[:i+1]
        M, targets = targeting_data[i]
        for j, theta in enumerate(M @ thetas[i]):
            yield cirq.Ry(rads=theta)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])
        for j, phi in enumerate(M @ phis[i]):
            yield cirq.Rz(rads=phi)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])

def ansatz_circuit(ket):
    params = ansatz_angles_to_params(*ket_to_ansatz_angles(ket))
    def __ansatz__(q, conjugate=False):
        yield __ansatz_circuit__(q, params, conjugate=conjugate)
    return __ansatz__

####################################################################################

def Z_d(d, qubits, aux, k=1):
    """Z acting on the first d basis vectors of n qubits. Requires two auxilliary qubits. Note d <= 2^{n-1}"""
    extended_qubits = [aux[0]] + qubits
    yield from Z(extended_qubits, k=k)
    yield from Z(extended_qubits, k=-d)
    yield from qft(extended_qubits, inverse=True)
    yield cirq.CNOT(extended_qubits[0], aux[1])
    yield from qft(extended_qubits)
    yield from [op.controlled_by(aux[1]) for op in Z(extended_qubits, k=d)]
    yield from Z(extended_qubits, k=-k)
    yield from qft(extended_qubits, inverse=True)
    yield cirq.X(extended_qubits[0])
    yield cirq.CNOT(extended_qubits[0], aux[1])
    yield cirq.X(extended_qubits[0])
    yield from qft(extended_qubits)
    yield from Z(extended_qubits, k=k)

def X_d(d, qubits, aux, k=1):
    """X acting on the first d basis vectors of n qubits. Requires two auxilliary qubits. Note d <= 2^{n-1}"""
    extended_qubits = [aux[0]] + qubits
    yield from qft(extended_qubits)
    yield from Z_d(d, qubits, aux, k=k)
    yield from qft(extended_qubits, inverse=True)

####################################################################################

def CZ_d(d, control_qubits, target_qubits, aux, inverse=False):
    """CZ acting on the first d basis vectors of two pairs of n qubits. Requires two auxilliary qubits. Note d <= 2^{n-1}"""
    if inverse:
        yield from cirq.inverse(list(CZ_d(d, control_qubits, target_qubits, aux)))
        return
    for i, control_qubit in enumerate(control_qubits):
        k = 2**(len(control_qubits) - 1 - i)
        yield from [op.controlled_by(control_qubit) for op in Z_d(d, target_qubits, aux, k=k)]

def CX_d(d, control_qubits, target_qubits, aux, inverse=False):
    """CX acting on the first d basis vectors of two pairs of n qubits. Requires two auxilliary qubits. Note d <= 2^{n-1}"""
    extended_target_qubits = [aux[0]] + target_qubits
    yield from qft(extended_target_qubits)
    yield from CZ_d(d, control_qubits, target_qubits, aux, inverse=inverse)
    yield from qft(extended_target_qubits, inverse=True)

####################################################################################

def qft_d(d, qubits, inverse=False):
    """QFT acting on the first d basis vectors of two pairs of n qubits."""
    if inverse:
        yield from cirq.inverse(qft_d(d, qubits))
        return
    n = len(qubits)
    F = np.array([[np.exp(2*np.pi*1j*i*j/d) for j in range(d)] for i in range(d)])/np.sqrt(d)
    Fd_gate = cirq.MatrixGate(sc.linalg.block_diag(F, np.eye(2**n - d)), name=f"DFT({d})")
    yield from cirq.decompose(cirq.Circuit((Fd_gate.on(*qubits))))