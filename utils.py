import collections
from functools import reduce

import numpy as np
from numpy.linalg import matrix_power as mpow

np.set_printoptions(precision=3, suppress=True)

def kron(*A):
    return reduce(np.kron, A)

def rand_ket(d):
    ket = np.random.randn(d) + 1j*np.random.randn(d)
    return ket/np.linalg.norm(ket)

# Construct WH operators
def wh_operators(d):
    w = np.exp(2*np.pi*1j/d)
    Z = np.diag(np.array([w**i for i in range(d)]))
    F = np.array([[w**(i*j) for j in range(d)] for i in range(d)])/np.sqrt(d)
    X = F.conj().T @ Z @ F
    D = np.array([[mpow(X, i) @ mpow(Z, j)  for j in range(d)] for i in range(d)])
    return locals()

# Generate WH-POVM elements from a fiducial ket
def wh_povm(phi):
	d = phi.shape[0]
	D = wh_operators(d)["D"]
	Pi = np.outer(phi, phi.conj())
	return np.array([D[a].conj().T @ Pi @ D[a] for a in np.ndindex(d,d)])/d

# Construct a d=4 SIC fiducial ket
def d4_fiducial_ket():
    H = np.array([[1,1],[1,-1]])/np.sqrt(2)
    P = np.diag([1, np.exp(1j*np.pi*(-1/4)), np.exp(1j*np.pi*(1/4)), np.exp(1j*np.pi*(1/2))]) 
    return np.kron(H, np.eye(2)) @ P @ (np.array([np.sqrt(2 + np.sqrt(5)), 1, 1, 1])/np.sqrt(5 + np.sqrt(5)))

def get_gate_counts(circuit):
    all_gate_types = [type(op.gate) for op in circuit.all_operations()]
    type_counts = collections.Counter(all_gate_types)
    print("--- Gate Counts (by type) ---")
    for gate_type, count in type_counts.items():
        print(f"{gate_type.__name__}: {count}")

# If we have WH-POVM probabilities (a d^2 vector), reorder them
# from the convention D^\dag \Pi D to D \Pi D^\dag (and vice versa)
def change_conjugate_convention(p):
    d = int(np.sqrt(p.shape[0]))
    idx_order = [0] +list(range(1, d))[::-1]
    return p.reshape(d,d)[np.ix_(idx_order, idx_order)].flatten()