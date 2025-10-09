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

def test_Xd_and_CXd(n=3):
	d = 2**n
	phi = rand_ket(d)
	ansatz_preparation = ansatz_circuit(phi)

	system_qubits = cirq.LineQubit.range(n)
	ancilla_qubits = cirq.LineQubit.range(n, 2*n)
	aux = [cirq.NamedQubit("aux1"), cirq.NamedQubit("aux2")]
	sim = cirq.Simulator()

	circ1 = cirq.Circuit((ansatz_preparation(system_qubits, conjugate=False),
						  X(system_qubits)))
	circ2 = cirq.Circuit((ansatz_preparation(system_qubits, conjugate=False),
						  X_d(d, system_qubits, aux, k=1)))

	res1, res2 = sim.simulate(circ1),sim.simulate(circ2)
	rho1, rho2= res1.density_matrix_of(system_qubits), res2.density_matrix_of(system_qubits)
	assert np.allclose(rho1, rho2)

	circ1 = cirq.Circuit((ansatz_preparation(ancilla_qubits, conjugate=True),
						  ansatz_preparation(system_qubits, conjugate=False),
						  CX(ancilla_qubits, system_qubits)))
	circ2 = cirq.Circuit((ansatz_preparation(ancilla_qubits, conjugate=True),
						  ansatz_preparation(system_qubits, conjugate=False),
						  CX_d(d, ancilla_qubits, system_qubits, aux)))
	res1, res2 = sim.simulate(circ1), sim.simulate(circ2)
	rho1, rho2 = res1.density_matrix_of(ancilla_qubits+system_qubits),\
			  	 res2.density_matrix_of(ancilla_qubits+system_qubits)
	assert np.allclose(rho1, rho2)

def test_simple_wh_povm_d(n=3, d_s=3):
	d = 2**n
	phi = load_sic_fiducial(d_s)
	ket = rand_ket(d_s)
	embedded_phi = np.concatenate([phi, np.zeros(d-d_s)])
	embedded_ket = np.concatenate([ket, np.zeros(d-d_s)])
	prepare_fiducial = ansatz_circuit(embedded_phi)
	prepare_system = ansatz_circuit(embedded_ket)

	system_qubits = cirq.LineQubit.range(n)
	ancilla_qubits = cirq.LineQubit.range(n, 2*n)
	aux = [cirq.NamedQubit("aux1"), cirq.NamedQubit("aux2")]
	sim = cirq.Simulator()

	circ = cirq.Circuit((prepare_system(system_qubits, conjugate=False),
						prepare_fiducial(ancilla_qubits, conjugate=True),
						CX_d(d_s, ancilla_qubits, system_qubits, aux, inverse=True),
						qft_d(d_s, ancilla_qubits, inverse=True)))
	res = sim.simulate(circ)
	total_p = np.diag(res.density_matrix_of(system_qubits+ancilla_qubits)).real
	p = mod_d_probabilities(total_p, d_s, n, 2)

	E = wh_povm(phi)
	p2 = change_conjugate_convention(np.array([ket.conj() @ e @ ket for e in E]).real)
	assert np.allclose(p, p2)