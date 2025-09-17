from utils import *
from circuits import *

def check_wh_identities(d):
	WH = wh_operators(d)
	D, w, X, Z = WH["D"], WH["w"], WH["X"], WH["Z"]
	assert np.all([np.allclose(D[a], w**(-a[0]*a[1])*mpow(Z, a[1]) @ mpow(X, a[0])) for a in np.ndindex(d,d)])
	assert np.all([np.allclose(D[a].conj().T, w**(a[0]*a[1])*D[-a[0] % d, -a[1] % d]) for a in np.ndindex(d,d)])
	assert np.all([np.allclose(D[a] @ D[b], w**(a[1]*b[0])*D[(a[0]+b[0])%d, (a[1]+b[1])%d]) for b in np.ndindex(d,d) for a in np.ndindex(d,d)])
	assert np.all([np.allclose(D[b].conj().T @ D[a] @ D[b], w**(a[1]*b[0] - a[0]*b[1])*D[a]) for b in np.ndindex(d,d) for a in np.ndindex(d,d)])
	assert np.allclose(np.array([[(D[a].conj().T @ D[b]).trace() for b in np.ndindex(d,d)] for a in np.ndindex(d,d)]), d*np.eye(d**2))

def check_ak(d):
	WH = wh_operators(d)
	Z, X, F, D, w = WH["Z"], WH["X"], WH["F"], WH["D"], WH["w"]

	Pi_pos = np.array([np.diag(np.eye(d)[i]) for i in range(d)])
	Pi_mom = np.array([np.outer(F[:,i], F[:,i].conj()) for i in range(d)])
	
	# Construct AK unitary
	U1 = sum([kron(mpow(X, -k), np.eye(d), Pi_pos[k]) for k in range(d)])
	U2 = sum([kron(np.eye(d), mpow(X, -m), Pi_mom[m]) for m in range(d)])
	U = U2 @ U1

	# Check alternative unitary
	U_alt = sum([kron(Pi_mom[k], Pi_mom[m], D[-m,k%d]) for m in range(d) for k in range(d)])
	assert np.allclose(U, U_alt)

	phi = rand_ket(d)
	Pi = np.outer(phi, phi.conj())
	FPi = F.conj().T @ Pi
	gamma = np.array([w**(k*m)*FPi[m,k] for k in range(d) for m in range(d)])

	# Get gamma from the fiducial
	gamma2 = sum([kron(Pi_pos[j], mpow(Z, j)) for j in range(d)]) @ np.kron(np.eye(d), F.conj().T) @ np.kron(phi.conj(), phi)
	assert np.allclose(gamma, gamma2)

	# Check alternative initial ancilla state expression
	gamma_alt = d**(-3/2)*np.array([sum([w**(-a*m + b*k)*(D[a,b].conj().T @ Pi).trace()\
					 for b in range(d) for a in range(d)]) for k in range(d) for m in range(d)])
	assert np.allclose(gamma, gamma_alt)

	E = wh_povm(phi)
	ket = rand_ket(d)
	p = np.array([ket.conj() @ e @ ket for e in E]).real

	initial_state = kron(gamma, ket)
	final_state = U @ initial_state
	ak_p = np.array([final_state.conj() @ kron(Pi_pos[x], Pi_pos[y], np.eye(d)) @ final_state\
				  		for x in range(d) for y in range(d)]).real
	
	# Compare direct WH-POVM probabilities with AK probabilities
	assert np.allclose(p, ak_p)
	
	# Usual Kraus operators
	K = np.array([kron(np.eye(d)[i], np.eye(d)[j], np.eye(d)) @ U @ kron(gamma, np.eye(d)).T for i in range(d) for j in range(d)])
	E2 = np.array([k.conj().T @ k for k in K])
	assert np.allclose(E, E2)

	# Check AK sans final Fourier: we get the same POVM elements
	U_sans = sum([kron(np.eye(d), mpow(X, -k), Pi_pos[k]) for k in range(d)]) @ kron(np.eye(d), np.eye(d), F.conj().T) @ sum([kron(mpow(X, -k), np.eye(d), Pi_pos[k]) for k in range(d)])
	K_sans = np.array([kron(np.eye(d)[i], np.eye(d)[j], np.eye(d)) @ U_sans @ kron(gamma, np.eye(d)).T for i in range(d) for j in range(d)])
	E_sans = np.array([k.conj().T @ k for k in K_sans])
	assert np.allclose(E, E_sans)

	# We get a different Kraus update
	rho = np.outer(ket, ket.conj())
	rho0 = (K[0] @ rho @ K[0].conj().T)
	rho0 = rho0/rho0.trace()
	rho0_sans = (K_sans[0] @ rho @ K_sans[0].conj().T)
	rho0_sans = rho0_sans/rho0_sans.trace()
	assert not np.allclose(rho0, rho0_sans)

def check_d4_fiducial_ket():
	d = 4
	E = wh_povm(d4_fiducial_ket())
	P = np.array([[(a@b/b.trace()).trace() for b in E] for a in E]).real
	P_SIC = np.array([[(d*(1 if i == j else 0) + 1)/(d*(d+1)) for j in range(d**2)] for i in range(d**2)]).real
	assert np.allclose(P, P_SIC)

def check_characteristic_state(d):
    WH = wh_operators(d)
    D, X, F = WH["D"], WH["X"], WH["F"]
    Pi_pos = np.array([np.diag(np.eye(d)[i]) for i in range(d)])
    BigD = np.kron(F.conj().T, np.eye(d)) @ sum([kron(Pi_pos[j], mpow(X, -j)) for j in range(d)])
    phi = rand_ket(d)
    Pi = np.outer(phi, phi.conj().T)
    char_ket1 = BigD @ np.kron(phi.conj(), phi)
    char_ket2 = sum([(D[a,b].conj().T @ Pi).trace()*np.kron(np.eye(d)[b], np.eye(d)[a]) for a in range(d) for b in range(d)])/np.sqrt(d)
    assert np.allclose(char_ket1, char_ket2)
	
def check_qft(n):
	q = cirq.LineQubit.range(n)
	circuit = cirq.Circuit(F(q))
	s = cirq.Simulator()
	ket = rand_ket(2**n)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector
	assert np.allclose(final_ket, wh_operators(2**n)["F"] @ ket)

def check_shift(n):
	q = cirq.LineQubit.range(n)
	circuit = cirq.Circuit(X(q))
	s = cirq.Simulator()
	ket = rand_ket(2**n)
	final_ket = s.simulate(circuit, initial_state=ket).final_state_vector
	assert np.allclose(final_ket, wh_operators(2**n)["X"] @ ket)

def check_controlled_shift(n):
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

def check_fiducial_preparation():
	n = 2
	q = cirq.LineQubit.range(n)
	circ = cirq.Circuit(d4_fiducial(q))
	assert np.allclose(d4_fiducial_ket(), cirq.unitary(circ)[:,0])

	circ = cirq.Circuit(d4_fiducial(q, conjugate=True))
	assert np.allclose(d4_fiducial_ket().conj(), cirq.unitary(circ)[:,0])

def check_simple_wh(d):
	WH = wh_operators(d)
	X_, F_, D_ = WH["X"], WH["F"], WH["D"]

	phi = rand_ket(d)
	E = wh_povm(phi)

	psi = rand_ket(d)
	p = np.array([psi.conj() @ e @ psi for e in E]).real
	p = change_conjugate_convention(p) # From D^\dag \Pi D to D \Pi D^\dag

	CX_ = sum([np.kron(mpow(X_, -j), np.outer(np.eye(d)[j], np.eye(d)[j])) for j in range(d)])
	V = np.kron(np.eye(d), F_.conj().T) @ CX_ 
	state = V @ np.kron(psi, phi.conj())
	p2 = abs(state)**2
	assert np.allclose(p, p2)

	V2 = sum([np.outer(np.kron(np.eye(d)[a], np.eye(d)[b]), D_[a,b].flatten().conj()) for a in range(d) for b in range(d)])/np.sqrt(d)
	assert np.allclose(V, V2)

	V3 = np.array([D_[a,b].flatten().conj() for a in range(d) for b in range(d)])/np.sqrt(d)
	assert np.allclose(V, V3)