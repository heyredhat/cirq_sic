import collections
from functools import reduce
import numpy as np

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

