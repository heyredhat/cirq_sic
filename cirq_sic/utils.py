import collections
from functools import reduce
import numpy as np
import string

import cirq 

sigma_x = np.array([[0,1], [1,0]])
sigma_y = np.array([[0,-1j], [1j, 0]]) 
sigma_z = np.array([[1,0], [0,-1]])

def kron(*A):
    """Tensor lots of things together."""
    return reduce(np.kron, A)

def rand_ket(d):
    """Random d-dimensional normalized complex vector."""
    ket = np.random.randn(d) + 1j*np.random.randn(d)
    return ket/np.linalg.norm(ket)

def ptrace(rho, over, dims):
    """Partial trace of a density matrix with ket dimensions dims over indices over."""
    indices = list(string.ascii_lowercase[:len(dims)*2])
    for o in over:
        indices[o+len(dims)] = indices[o]
    return np.einsum("".join(indices), rho.reshape(dims*2))

def pad(x, d):
    return np.concatenate([x, np.zeros(d-len(x))])

def get_gate_counts(circuit):
    """Get gate counts for a cirq circuit."""
    all_gate_types = [type(op.gate) for op in circuit.all_operations()]
    type_counts = collections.Counter(all_gate_types)
    print("--- Gate Counts (by type) ---")
    for gate_type, count in type_counts.items():
        print(f"{gate_type.__name__}: {count}")

def symmetrize(M, T=100):
    """Obtain a stochastic symmetric matrix by a variant of Sinkhorn's algorithm."""
    for t in range(T):
        M = (M + M.T)/2
        M = M/np.sum(M, axis=0)
    return M

def nonneg_projection(p):
    """Project a vector to the probability simplex by setting negatives to zero and renormalizing."""
    p_fixed = p[:]
    p_fixed[p < 0] = 0
    p_fixed = p_fixed/sum(p_fixed)
    return p_fixed

def dirac(state_vector):
    """n qubit state vector in Dirac notation."""
    n = int(np.log2(len(state_vector)))
    basis_states = []
    for i, amp in enumerate(state_vector):
        if abs(amp) > 0.001:
            bin_str = format(i, f'0{n}b')
            print("%s: %.2f+%.2fj: %.3f" % (bin_str, amp.real, amp.imag, abs(amp)**2))
            basis_states.append((bin_str, amp))
    return basis_states

def mod_d_outcome_mask(d, n, m):
    """When working on computations mod d encoded in n-qubits, with m groups of n-qubits."""
    d_b = 2**n
    return sum([kron(*[np.eye(d_b, dtype=int)[i] for i in ind]) for ind in np.ndindex(*[d_b]*m) if np.all(np.array(ind) < d)])

def mod_d_probabilities(p, d, n, m):
    return p[np.where(mod_d_outcome_mask(d, n, m)==1)]