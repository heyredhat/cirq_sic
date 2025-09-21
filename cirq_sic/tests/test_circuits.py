from cirq_sic import *
import numpy as np

d = 4
n = 2

def test_qft(n=2):
	q = cirq.LineQubit.range(n)
	circuit = cirq.Circuit(F(q))
	s = cirq.Simulator()
	ket = rand_ket(2**n)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector
	assert np.allclose(final_ket, wh_operators(2**n)["F"] @ ket)

def test_shift(n=2):
	q = cirq.LineQubit.range(n)
	circuit = cirq.Circuit(X(q))
	s = cirq.Simulator()
	ket = rand_ket(2**n)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector
	assert np.allclose(final_ket, wh_operators(2**n)["X"] @ ket)

def test_controlled_shift(n=2):
	d = 2**n
	t = cirq.LineQubit.range(n)
	c = cirq.LineQubit.range(n, 2*n)
	circuit = cirq.Circuit(CX(c, t))

	s = cirq.Simulator()
	ket = rand_ket(d**2)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector

	X_ = wh_operators(d)["X"]
	CX_ = sum([kron(mpow(X_, k), np.outer(np.eye(d)[k], np.eye(d)[k])) for k in range(d)])
	assert np.allclose(final_ket, CX_ @ ket)

def test_fiducial_preparation():
	n = 2
	q = cirq.LineQubit.range(n)
	circ = cirq.Circuit(d4_sic_fiducial(q))
	assert np.allclose(d4_sic_fiducial_ket(), cirq.unitary(circ)[:,0])

	circ = cirq.Circuit(d4_sic_fiducial(q, conjugate=True))
	assert np.allclose(d4_sic_fiducial_ket().conj(), cirq.unitary(circ)[:,0])
	
def test_qudit_basis_state():
	n = 3
	d = 2**n
	q = cirq.LineQubit.range(n)
	for i in range(d):
		circ = cirq.Circuit((qudit_basis_state(q, i)))
		ket = cirq.Simulator().simulate(circ, qubit_order=q).final_state_vector
		assert np.allclose(ket, np.eye(d)[i])

def test_ansatz_circuit(n=3):
	d = 2**n
	ket = rand_ket(d)
	q = cirq.LineQubit.range(n)
	prepare_ansatz = ansatz_circuit(ket)
	circ = cirq.Circuit((prepare_ansatz(q, conjugate=True)))
	final_ket = cirq.Simulator().simulate(circ, qubit_order=q).final_state_vector
	assert np.allclose(final_ket, ket.conj())