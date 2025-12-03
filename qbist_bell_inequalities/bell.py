import numpy as np
np.set_printoptions(precision=3, suppress=True)

from itertools import product
from functools import reduce

import cvxpy as cp

##################################################################

def kron(*A):
    return reduce(np.kron, A)

##################################################################

def construct_joint_reference(reference_measurements, reference_states):
    """Given reference measurements and reference states for each party, construct the effective joint reference device."""
    d = reference_measurements[0].shape[-1]
    n_parties = reference_measurements.shape[0]
    # List of P matrices for each party
    P = np.array([[[(a@b).trace() for b in reference_states[i]] for a in reference_measurements[i]] for i in range(n_parties)]).real
    # List of Phi matrices for each party
    Phi = np.array([np.linalg.inv(_) for _ in P])
    # Joint reference POVM elements
    R = np.array([kron(*[reference_measurements[i][j] for i, j in enumerate(idx)]) for idx in np.ndindex(*[reference_measurements.shape[1]]*n_parties)])
    # Joint reference states
    S = np.array([kron(*[reference_states[i][j] for i, j in enumerate(idx)]) for idx in np.ndindex(*[reference_measurements.shape[1]]*n_parties)])
    # Joint Phi matrix
    phi = kron(*Phi).flatten()
    return locals()

def construct_projectors(observables, labels=False):
    """Observables is a list of length n_parties: it specifies a list of observables for each party
    in a Bell scenario. Constructs the corresponding projectors onto the eigenvectors of each observable."""
    eigvals, eigvecs = np.linalg.eigh(observables) 
    projectors = np.einsum('...ik,...jk->...kij', eigvecs, eigvecs.conj())
    if labels:
        labels = np.array([[[f"A{i}: o{k} of O{j}" for k in range(eigvals.shape[-1])] for j in range(observables.shape[1])] for i in range(observables.shape[0])])
        return projectors, labels
    return projectors

def construct_tensor_projectors(projectors, labels=None):
    """Given projectors (which party, which observable, which outcome), construct the tensor product observables over parties."""
    d = projectors.shape[-1]
    n_parties = projectors.shape[0]
    n_measurements = projectors.shape[1]
    tensor_projectors = np.array([reduce(np.kron, (proj[idx] for proj, idx in zip(projectors, outcome_indices)))\
                                    for outcome_indices in product(*[range(n) for n in [d]*n_parties])])\
                                        .reshape(-1, d**n_parties, d**n_parties)
    if type(labels) != type(None):
        label_blocks = []
        for meas_choice in product(range(n_measurements), repeat=n_parties):
            outcome_labels = labels[np.arange(n_parties), meas_choice]
            block = [' ⊗ '.join(outcome_labels[p][outcome_choice[p]] for p in range(n_parties))
                for outcome_choice in product(range(d), repeat=n_parties)]
            label_blocks.append(block)
        tensor_labels = np.array(label_blocks).reshape(-1)
        return tensor_projectors, tensor_labels
    return tensor_projectors

def construct_p(observables, state, labels=False):
    """Construct joint probability distribution over observables (for each party, a list of observables) given the joint entangled state."""
    if labels:
        projectors, proj_labels = construct_projectors(observables, labels=labels)
        tensor_projectors, tensor_labels = construct_tensor_projectors(projectors, labels=proj_labels)
        p = np.einsum("ijk, kj", tensor_projectors, state)
        return p, tensor_labels
    else:
        return np.einsum("ijk, kj", construct_tensor_projectors(construct_projectors(observables)), state)

def construct_T(state, R, projectors, reference_states):
    """Given a choice of state, a joint reference POVM R, a specification of projectors for observables for each party, and reference states for each party,
     constructs the T matrix."""
    d = projectors.shape[-1]
    n_parties = projectors.shape[0]
    PR = np.einsum("ijk, kj", R, state).real
    PER = np.array([np.einsum("ijkl, mlk", projectors[i], reference_states[i]) for i in range(n_parties)]).real
    JPER = np.array([reduce(np.kron, (per[idx] for per, idx in zip(PER, outcome_indices)))\
                            for outcome_indices in product(*[range(n) for n in [d]*n_parties])])\
                                .reshape(-1, d**(2*n_parties))
    return kron(JPER, PR).reshape(-1, PR.shape[0]**2)

def quantumness_bound(bell_functional, classical_bound, T, phi):
    """Returns the bound on ||I - Phi||."""
    p = T @ phi
    T_singular_values = np.linalg.svd(T, compute_uv=False)
    Delta = bell_functional @ p - classical_bound
    bound = Delta/(np.max(T_singular_values)*np.linalg.norm(bell_functional))
    return bound

##################################################################

X, Y, Z = np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])

def chsh_inequality():
    v = np.array([-1,1])
    s = np.array([[1,1],[1,-1]])
    bell_functional = kron(s.flatten(), v, v)
    classical_bound = 2
    return (bell_functional, classical_bound)

def example_chsh_saturator():
    observables = np.array([[Z, X], [(Z+X)/np.sqrt(2), (Z-X)/np.sqrt(2)]])
    ket = np.array([1,0,0,1])/np.sqrt(2)
    return observables, np.outer(ket, ket.conj())

##################################################################

def split_into_segments(s, n):
    """Splits a list s into n segments."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(s) % n != 0:
        raise ValueError("length not divisible by n")
    size = len(s) // n
    return [s[i:i+size] for i in range(0, len(s), size)]

def construct_deterministic_behaviors(n_parties, n_measurements, n_outcomes):
    """Construct a list of the deterministic behaviors in a Bell scenario with
       n_parties, n_measurements per party, and where the number of outcomes of
       each measurement is n_outcomes = [num_outcomes_for_measurement1, num_outcomes_for_measurement2, ...]."""
    lambdas = list(product(*[range(_) for _ in n_outcomes]*n_parties))
    deterministic_behaviors = []
    for assignments in lambdas:
        party_assignments = split_into_segments(assignments, n_parties)
        behavior = [1 if np.all([o == party_assignments[i][setting_idx[i]] for i, o in enumerate(outcome_idx)]) else 0 \
            for setting_idx in np.ndindex(*[n_measurements]*n_parties)
                for outcome_idx in product(*[range(_) for _ in n_outcomes])]
        deterministic_behaviors.append(behavior)
    return np.array(deterministic_behaviors)

def construct_bell_inequality(n_parties, n_measurements, n_outcomes, p, dichotomous=False):
    """Returns a Bell inequality which witnesses the nonclassicality of a joint probability distribution p."""
    n = np.prod(n_outcomes)*n_measurements*n_parties
    deterministic_behaviors = construct_deterministic_behaviors(n_parties, n_measurements, n_outcomes)
    classical_bound_var = cp.Variable()
    if dichotomous:
        b = cp.Variable(n, boolean=True)
        bell_functional_var = 2*b - 1
    else:
        bell_functional_var = cp.Variable(n)
    problem = cp.Problem(cp.Maximize(p @ bell_functional_var - classical_bound_var),\
                        [deterministic_behaviors @ bell_functional_var - classical_bound_var <= 0,
                        p @ bell_functional_var - classical_bound_var <= 1])
    value = problem.solve()
    bell_functional = bell_functional_var.value
    classical_bound = classical_bound_var.value
    return (bell_functional, classical_bound)

##################################################################

def affine_dimension(points, tol=1e-9):
    """
    Compute the affine dimension of a set of points in R^N,
    given as rows of an array of shape (M, N).
    """
    points = np.asarray(points, dtype=float)
    M, N = points.shape
    if M <= 1:
        return 0
    p0 = points[0, :]
    A = points[1:, :] - p0[None, :]
    return np.linalg.matrix_rank(A, tol)

def is_facet_inequality(D, s, c, tol=1e-7):
    """
    Check if the inequality s·p <= c is facet-defining for the polytope
    whose vertices are the rows of D.

    Parameters
    ----------
    D : np.ndarray, shape (L, N)
        Rows are deterministic behaviors d_lambda.
    s : np.ndarray, shape (N,)
        Coefficients of the Bell inequality.
    c : float
        Bound in the inequality s·p <= c.
    tol : float
        Numerical tolerance.

    Returns
    -------
    is_facet : bool
        True if the inequality is facet-defining.
    info : dict
        Extra info: polytope dimension, face dimension,
        # of saturating vertices, and their indices.
    """
    D = np.asarray(D, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1)

    # Values of s·d_lambda for each deterministic vertex (row)
    vals = D @ s  # shape (L,)

    # Deterministic points that saturate the inequality
    sat_mask = np.abs(vals - c) <= tol
    sat_indices = np.where(sat_mask)[0]

    if len(sat_indices) == 0:
        return False, {
            "reason": "No deterministic vertices saturate the inequality.",
            "dim_polytope": None,
            "dim_face": None,
            "num_saturating": 0,
            "saturating_indices": []
        }

    D_sat = D[sat_indices, :]   # rows that saturate

    # Affine dimension of the whole local polytope
    dim_poly = affine_dimension(D)

    # Affine dimension of the face defined by the inequality
    dim_face = affine_dimension(D_sat)

    is_facet = (dim_face == dim_poly - 1)

    info = {
        "dim_polytope": dim_poly,
        "dim_face": dim_face,
        "num_saturating": len(sat_indices),
        "saturating_indices": sat_indices
    }

    return is_facet, info
