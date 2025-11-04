from functools import reduce
import numpy as np
import string

####################################################################################

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

def rand_dm(d, r=1):
    """Random dxd density matrix with rank r."""
    A = np.random.randn(d, r) + 1j*np.random.randn(d, r)
    rho = A @ A.conj().T
    return rho/rho.trace()

def ptrace(rho, over, dims):
    """Partial trace of a density matrix with ket dimensions dims over indices over."""
    indices = list(string.ascii_lowercase[:len(dims)*2])
    for o in over:
        indices[o+len(dims)] = indices[o]
    return np.einsum("".join(indices), rho.reshape(dims*2))

####################################################################################

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

####################################################################################

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

####################################################################################

def pad(x, d):
    """Pads a vector with 0's on the right so it is of length d."""
    return np.concatenate([x, np.zeros(int(d-len(x)))])

def mod_d_outcome_mask(d, n, m):
    """When working on computations mod d encoded in n-qubits, with m groups of n-qubits."""
    d_b = 2**n
    return sum([kron(*[np.eye(d_b, dtype=int)[i] for i in ind]) for ind in np.ndindex(*[d_b]*m) if np.all(np.array(ind) < d)])

def mod_d_probabilities(p, d, n, m):
    """Given a probability vector p, for a dimension d computation, which is encoded in m groups of n qubits, return just the relevant probabilities."""
    return p[np.where(mod_d_outcome_mask(d, n, m)==1)]

####################################################################################

def geodesic_interpolator(psi_start, psi_target, atol=1e-12):
    """
    Build psi(t) that moves along the Fubini–Study geodesic from psi_start to psi_target.

    Parameters
    ----------
    psi_start, psi_target : array_like
        Column vectors (shape (d,) or (d,1)). They will be normalized internally.
    atol : float
        Angular tolerance treating states as identical.

    Returns
    -------
    psi : callable
        psi(t) with t ∈ [0, 1], giving a normalized vector.
    """
    psi_start = np.asarray(psi_start, dtype=complex).reshape(-1)
    psi_target = np.asarray(psi_target, dtype=complex).reshape(-1)

    # Normalize both inputs
    psi_start = psi_start / np.linalg.norm(psi_start)
    psi_target = psi_target / np.linalg.norm(psi_target)

    # Align global phase of the target so overlap is real and non-negative
    overlap = np.vdot(psi_start, psi_target)
    phase = np.angle(overlap)
    aligned = np.exp(-1j * phase) * psi_target

    c = np.real(np.vdot(psi_start, aligned))
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < atol:
        def psi(t):
            return psi_start.copy()
        return psi

    sin_theta = np.sin(theta)
    eta = (aligned - c * psi_start) / sin_theta  # unit vector orthogonal to psi_start

    def psi(t):
        t = np.asarray(t, dtype=float)
        return np.cos(t * theta) * psi_start + np.sin(t * theta) * eta

    return psi

####################################################################################

def merge_dicts(dicts):
    """Helper function to merge dictionaries. """
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged