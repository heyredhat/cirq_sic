from cirq_sic import *
import numpy as np

d = 4
n = 2

def test_qudit_basis_state():
	n = 3
	d = 2**n
	q = cirq.LineQubit.range(n)
	for i in range(d):
		circ = cirq.Circuit((qudit_basis_state(q, i)))
		ket = cirq.Simulator().simulate(circ, qubit_order=q).final_state_vector
		assert np.allclose(ket, np.eye(d)[i])

def test_qft(n=2):
	q = cirq.LineQubit.range(n)
	circuit = cirq.Circuit((qft(q, inverse=True)))
	s = cirq.Simulator()
	ket = rand_ket(2**n)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector
	assert np.allclose(final_ket, wh_operators(2**n)["F"].conj().T @ ket)

def test_clock(n=2, k=3):
	ket = rand_ket(2**n)
	q = cirq.LineQubit.range(n)
	circ = cirq.Circuit((Z(q, k=k)))
	sim = cirq.Simulator()
	assert np.allclose(sim.simulate(circ, initial_state=ket).state_vector(), mpow(wh_operators(2**n)["Z"], k) @ ket)

def test_shift(n=2, k=3):
	ket = rand_ket(2**n)
	q = cirq.LineQubit.range(n)
	circ = cirq.Circuit((X(q, k=k)))
	sim = cirq.Simulator()
	assert np.allclose(sim.simulate(circ, initial_state=ket).state_vector(), mpow(wh_operators(2**n)["X"], k) @ ket)

def test_controlled_shift(n=2):
	d = 2**n
	t = cirq.LineQubit.range(n)
	c = cirq.LineQubit.range(n, 2*n)
	circuit = cirq.Circuit(CX(c, t, inverse=True))

	s = cirq.Simulator()
	ket = rand_ket(d**2)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector

	X_ = wh_operators(d)["X"]
	CX_ = sum([kron(mpow(X_, -k), np.outer(np.eye(d)[k], np.eye(d)[k])) for k in range(d)])
	assert np.allclose(final_ket, CX_ @ ket)

def test_fiducial_preparation():
	n = 2
	q = cirq.LineQubit.range(n)
	circ = cirq.Circuit(d4_sic_fiducial(q))
	assert np.allclose(d4_sic_fiducial_ket(), cirq.unitary(circ)[:,0])

	circ = cirq.Circuit(d4_sic_fiducial(q, conjugate=True))
	assert np.allclose(d4_sic_fiducial_ket().conj(), cirq.unitary(circ)[:,0])
	
def test_ansatz_circuit(n=3):
	d = 2**n
	ket = rand_ket(d)
	q = cirq.LineQubit.range(n)
	prepare_ansatz = ansatz_circuit(ket)
	circ = cirq.Circuit((prepare_ansatz(q, conjugate=True)))
	final_ket = cirq.Simulator().simulate(circ, qubit_order=q).final_state_vector
	assert np.allclose(final_ket, ket.conj())

def test_simple_wh_povm(n=3):
	d = 2**n

	ket = rand_ket(d)
	prepare_system = ansatz_circuit(ket)

	phi = load_sic_fiducial(d)
	prepare_fiducial = ansatz_circuit(phi)

	system_qubits = cirq.LineQubit.range(n)
	ancilla_qubits = cirq.LineQubit.range(n, 2*n)

	circ = cirq.Circuit((prepare_system(system_qubits),
						 simple_wh_povm(system_qubits,\
										ancilla_qubits,\
										prepare_fiducial=prepare_fiducial, measure=False)))
	sim = cirq.Simulator()
	p = np.diag(sim.simulate(circ).density_matrix_of(system_qubits+ancilla_qubits)).real

	E = wh_povm(phi)
	p2 = change_conjugate_convention(np.array([ket.conj() @ e @ ket for e in E]).real)
	assert np.allclose(p, p2)

def test_arthurs_kelly(n=3):
	d = 2**n

	ket = rand_ket(d)
	prepare_system = ansatz_circuit(ket)

	phi = load_sic_fiducial(d)
	prepare_fiducial = ansatz_circuit(phi)

	system_qubits = cirq.LineQubit.range(n)
	ancilla1_qubits = cirq.LineQubit.range(n, 2*n)
	ancilla2_qubits = cirq.LineQubit.range(2*n, 3*n)

	circ = cirq.Circuit((prepare_system(system_qubits),
						 arthurs_kelly(system_qubits,\
									   ancilla1_qubits, ancilla2_qubits,
									   prepare_fiducial=prepare_fiducial, measure=False)))
	sim = cirq.Simulator()
	p = np.diag(sim.simulate(circ).density_matrix_of(ancilla1_qubits+ancilla2_qubits)).real

	E = wh_povm(phi)
	p2 = np.array([ket.conj() @ e @ ket for e in E]).real
	assert np.allclose(p, p2)
