import numpy as np
np.set_printoptions(precision=3, suppress=True)

from itertools import product
from functools import reduce

import cvxpy as cp

##################################################################

def kron(*A):
    return reduce(np.kron, A)

def transpose_povm(E):
    return np.array([e.T for e in E])

##################################################################

def construct_deterministic_behaviors(w, return_lambdas=False):
    """
    Given a Bell scenario specified by a list w, construct all deterministic
    behaviors as rows of an array.

    Parameters
    ----------
    w : list of lists of ints
        w[i] is a list of outcome counts for party i.
        Example:
            w = [
                [d, d**2, 3],   # party 0: 3 measurements, with d, d^2, 3 outcomes
                [2, d**3],      # party 1: 2 measurements, with 2, d^3 outcomes
                [d, d, d]       # party 2: 3 measurements, all with d outcomes
            ]

    Returns
    -------
    behaviors : np.ndarray, shape (num_lambdas, num_probabilities)
        Each row is a deterministic behavior p(a_0,...,a_{N-1} | x_0,...,x_{N-1}),
        flattened in lex order: first over all settings (x_0,...,x_{N-1}),
        then over all joint outcomes consistent with that setting.
    lambdas : list
        List of deterministic assignments. Each element is a tuple
        (lambda_0, ..., lambda_{N-1}), where lambda_i is a tuple of length
        len(w[i]) giving the predetermined outcome for each measurement of party i.
    """
    # number of parties
    n_parties = len(w)
    if n_parties == 0:
        raise ValueError("w must describe at least one party.")

    # 1) Build the space of deterministic assignments for each party
    #    For party i, a deterministic assignment is a tuple:
    #        lambda_i = (a_i^0, a_i^1, ..., a_i^{M_i-1})
    #    where a_i^m in range(w[i][m])
    party_assignment_spaces = []
    for party_outcomes in w:
        if len(party_outcomes) == 0:
            raise ValueError("Each party must have at least one measurement.")
        ranges_for_party = [range(o) for o in party_outcomes]
        # All tuples of predetermined outcomes for this party
        party_assignment_spaces.append(list(product(*ranges_for_party)))

    # Global deterministic lambdas: one assignment per party
    lambdas = list(product(*party_assignment_spaces))
    # lambdas[k] is a tuple (lambda_0, ..., lambda_{N-1})
    # with lambda_i a tuple of predetermined outcomes for party i.

    # 2) Enumerate all measurement settings:
    #    For party i, measurement index is in range(len(w[i]))
    settings_ranges = [range(len(party_outcomes)) for party_outcomes in w]
    all_settings = list(product(*settings_ranges))

    # 3) Compute the total length of the behavior vector (for sanity)
    expected_length = 0
    for setting in all_settings:  # setting = (x_0, ..., x_{N-1})
        num_joint_outcomes = 1
        for party, meas in enumerate(setting):
            num_joint_outcomes *= w[party][meas]
        expected_length += num_joint_outcomes

    # 4) Build the deterministic behaviors
    behaviors = []

    for lam in lambdas:
        # lam[i] is a tuple of predetermined outcomes for party i:
        # lam[i][m] = outcome that party i outputs when measuring m
        behavior = []

        # Loop over all measurement settings
        for setting in all_settings:
            # For this setting, party i chooses measurement setting[i]
            # and has w[i][setting[i]] possible outcomes.
            per_party_outcome_ranges = [
                range(w[party][setting[party]]) for party in range(n_parties)
            ]

            # Loop over all joint outcomes for this setting
            for outcome_tuple in product(*per_party_outcome_ranges):
                # outcome_tuple[party] is the output of party i
                # This deterministic strategy is 1 iff each party outputs its
                # predetermined outcome for the chosen measurement.
                is_this_lambda = all(
                    outcome_tuple[party] == lam[party][setting[party]]
                    for party in range(n_parties)
                )
                behavior.append(1 if is_this_lambda else 0)

        if len(behavior) != expected_length:
            raise RuntimeError(
                f"Internal error: behavior length {len(behavior)} "
                f"!= expected {expected_length}"
            )

        behaviors.append(behavior)
    
    behaviors = np.array(behaviors, dtype=int)
    return (behaviors, lambdas) if return_lambdas else behaviors

def construct_bell_inequality(w, p, dichotomous=False, return_problem=False):
    """Returns a Bell inequality which witnesses the nonclassicality of a joint probability distribution p."""
    n = len(p)
    deterministic_behaviors = construct_deterministic_behaviors(w)
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
    return (bell_functional, classical_bound) if not return_problem else (bell_functional, classical_bound, problem)

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

##################################################################

def construct_povms_from_observables(observables, labels=False):
    """Observables is a list of length n_parties: it specifies a list of observables for each party
    in a Bell scenario. Constructs the corresponding projectors onto the eigenvectors of each observable."""
    eigvals, eigvecs = np.linalg.eigh(observables) 
    projectors = np.einsum('...ik,...jk->...kij', eigvecs, eigvecs.conj())
    return projectors

##################################################################

def quantum_behavior_from_povms(E, rho, return_metadata=False):
    """
    Given a multipartite POVM specification and a global quantum state,
    compute the full behavior p(a_0,...,a_{n-1} | x_0,...,x_{n-1}).

    Parameters
    ----------
    E : list
        Nested list specifying local POVMs.
        E[party][measurement][outcome] is a numpy array (matrix) acting
        on the Hilbert space of that party.
        - party runs from 0 to n_parties-1
        - measurement index runs from 0 to n_meas(party)-1
        - outcome index runs from 0 to n_outcomes(party,measurement)-1

    rho : np.ndarray
        Global n-partite quantum state (density matrix) on the tensor product
        Hilbert space. Shape must be (D, D), where
        D = product over parties of local dimensions.

    Returns
    -------
    probs : np.ndarray, shape (num_probabilities,)
        Flattened probability distribution over all settings and outcomes
        in the following lexicographic order:

        - First over all settings (x_0,...,x_{n-1}) with:
            x_i in range(len(E[i]))
          ordered lexicographically in party index.

        - For each setting, over all joint outcomes (a_0,...,a_{n-1}),
          where a_i in range(len(E[i][x_i])), also in lexicographic order.

    meta : dict
        Metadata describing the ordering, with keys:
        - "settings": list of tuples (x_0,...,x_{n-1})
        - "outcome_ranges": list of lists; for each setting s,
              outcome_ranges[s] is a list [n_out0, n_out1, ..., n_out_{n-1}]
        - "n_parties": number of parties
        - "local_dims": list of local Hilbert space dimensions
    """
    # Ensure rho is a numpy array
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix (density operator).")

    n_parties = len(E)
    if n_parties == 0:
        raise ValueError("E must describe at least one party.")

    # 1) Infer local dimensions and sanity-check POVM shapes
    local_dims = []
    for party, party_povms in enumerate(E):
        if len(party_povms) == 0:
            raise ValueError(f"Party {party} must have at least one measurement.")
        # Take the first measurement and first outcome to infer dim
        first_meas = party_povms[0]
        if len(first_meas) == 0:
            raise ValueError(f"Party {party} measurement 0 has no outcomes.")
        first_elem = np.asarray(first_meas[0])
        if first_elem.ndim != 2 or first_elem.shape[0] != first_elem.shape[1]:
            raise ValueError(f"POVM element for party {party} is not a square matrix.")
        d_i = first_elem.shape[0]
        local_dims.append(d_i)

        # Check all POVM elements for this party have same shape
        for m, povm in enumerate(party_povms):
            for o, elem in enumerate(povm):
                elem = np.asarray(elem)
                if elem.shape != (d_i, d_i):
                    raise ValueError(
                        f"POVM element for party {party}, measurement {m}, "
                        f"outcome {o} has shape {elem.shape}, expected {(d_i, d_i)}."
                    )

    # 2) Check rho dimension
    D = 1
    for d_i in local_dims:
        D *= d_i
    if rho.shape != (D, D):
        raise ValueError(
            f"rho has shape {rho.shape}, but product of local dims is {D}."
        )

    # 3) Enumerate all settings
    settings_ranges = [range(len(party_povms)) for party_povms in E]
    settings_list = list(product(*settings_ranges))  # list of (x_0,...,x_{n-1})

    # 4) Precompute total length and outcome ranges per setting
    outcome_ranges_per_setting = []
    total_len = 0
    for setting in settings_list:
        # setting = (x_0,...,x_{n-1})
        outcome_counts = [len(E[party][setting[party]]) for party in range(n_parties)]
        outcome_ranges_per_setting.append(outcome_counts)
        num_joint_outcomes = 1
        for c in outcome_counts:
            num_joint_outcomes *= c
        total_len += num_joint_outcomes

    # Helper: Kronecker product of list of operators
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # 5) Compute probabilities
    probs = np.zeros(total_len, dtype=float)
    idx = 0  # current write position in probs

    for s_idx, setting in enumerate(settings_list):
        outcome_counts = outcome_ranges_per_setting[s_idx]
        outcome_ranges = [range(c) for c in outcome_counts]

        for outcome_tuple in product(*outcome_ranges):
            # Build global POVM operator = tensor product of local elements
            local_elements = [
                E[party][setting[party]][outcome_tuple[party]]
                for party in range(n_parties)
            ]
            M = kron_all(local_elements)  # global POVM element

            # Probability = Tr(rho M)
            p = np.trace(rho @ M)
            # Numerical noise: take real part and clip tiny negatives
            p_real = float(np.real_if_close(p))
            if p_real < 0 and p_real > -1e-12:
                p_real = 0.0
            probs[idx] = p_real
            idx += 1

    meta = {
        "settings": settings_list,
        "outcome_ranges": outcome_ranges_per_setting,
        "n_parties": n_parties,
        "local_dims": local_dims,
    }

    return (probs, meta) if return_metadata else probs

##################################################################

def test_ordering_alignment(w, n_tests=5, tol=1e-9):
    """
    Test that construct_deterministic_behaviors(w) and
    quantum_behavior_from_povms(E, rho) use the SAME ordering convention
    for (settings, outcomes).

    Parameters
    ----------
    w : list of lists of ints
        Scenario specification: w[party][measurement] = # outcomes.
    n_tests : int
        Number of deterministic strategies to test (at most the total number).
    tol : float
        Numerical tolerance for comparing probability vectors.

    Returns
    -------
    aligned : bool
        True if all tested deterministic behaviors match.
    details : dict
        Extra info, including indices tested and any mismatch info.
    """
    # 1) Get deterministic behaviors + their assignments
    D, lambdas = construct_deterministic_behaviors(w, return_lambdas=True)
    num_lambdas, num_probs = D.shape

    if num_lambdas == 0:
        raise ValueError("No deterministic behaviors for this specification w.")

    # Choose which lambdas to test (spread across the list if many)
    if n_tests >= num_lambdas:
        test_indices = list(range(num_lambdas))
    else:
        test_indices = np.linspace(0, num_lambdas - 1, n_tests, dtype=int).tolist()

    mismatches = []

    # Identity and zero in 1D Hilbert space
    I1 = np.array([[1.0]])
    Z1 = np.array([[0.0]])

    for idx in test_indices:
        lam = lambdas[idx]  # lam[party][measurement]

        # 2) Build POVMs E implementing this deterministic strategy in 1D
        E = []
        for party, party_outcomes in enumerate(w):
            party_povms = []
            for m, n_out in enumerate(party_outcomes):
                effects = []
                for k in range(n_out):
                    if k == lam[party][m]:
                        effects.append(I1)
                    else:
                        effects.append(Z1)
                party_povms.append(effects)
            E.append(party_povms)

        # 3) Global state rho = |1><1| in 1D
        rho = np.array([[1.0]])

        # 4) Compute probabilities via quantum_behavior_from_povms
        probs, meta = quantum_behavior_from_povms(E, rho, return_metadata=True)

        # 5) Compare with the deterministic vector from D
        det_vec = D[idx, :]

        if probs.shape[0] != num_probs:
            mismatches.append({
                "index": idx,
                "reason": f"Length mismatch: quantum {probs.shape[0]} vs deterministic {num_probs}"
            })
            continue

        if not np.allclose(probs, det_vec, atol=tol):
            mismatches.append({
                "index": idx,
                "reason": "Values differ",
                "deterministic": det_vec,
                "quantum": probs
            })

    aligned = (len(mismatches) == 0)

    details = {
        "aligned": aligned,
        "tested_indices": test_indices,
        "num_lambdas": num_lambdas,
        "num_probabilities": num_probs,
        "mismatches": mismatches
    }

    return aligned, details

##################################################################

def construct_joint_reference(reference_measurements, reference_states):
    """
    Given reference measurements and reference states for each party,
    construct the effective joint reference device and the vector phi.

    Parameters
    ----------
    reference_measurements : list
        reference_measurements[i][r] is a POVM element R_r^(i) (numpy array).
    reference_states : list
        reference_states[i][u] is a density matrix sigma_u^(i) (numpy array).

    Returns
    -------
    phi : np.ndarray, shape (prod_i (n_R_i * n_S_i),)
        Flattened vector of the Kronecker product of single-party Phi matrices.
    Phi_matrices : list
        Single-party Phi^(i) matrices, each of shape (n_R_i, n_S_i).
    P_matrices : list
        Single-party Born matrices P^(i), shape (n_R_i, n_S_i).
    joint_reference_measurement : list of np.ndarray
        Joint reference POVM elements, one per tuple (s_0,...,s_{n-1}).
    joint_reference_states : list of np.ndarray
        Joint reference states, one per tuple (u_0,...,u_{n-1}).
    meta : dict
        Contains:
        - "n_parties"
        - "n_R_per_party"
        - "n_S_per_party"
        - "ref_outcome_tuples": list of all (s_0,...,s_{n-1})
        - "ref_state_tuples":   list of all (u_0,...,u_{n-1})
    """
    n_parties = len(reference_measurements)
    if n_parties == 0:
        raise ValueError("Need at least one party.")

    P_matrices = []
    Phi_matrices = []
    n_R_per_party = []
    n_S_per_party = []

    # 1) Single-party Born matrices and pseudoinverses
    for i in range(n_parties):
        R_i = reference_measurements[i]
        S_i = reference_states[i]
        n_R = len(R_i)
        n_S = len(S_i)
        if n_R == 0 or n_S == 0:
            raise ValueError(f"Party {i} must have at least one reference POVM element and state.")

        # Build P^(i)_{r u} = Tr(R_r sigma_u)
        P = np.zeros((n_R, n_S), dtype=float)
        for r, R in enumerate(R_i):
            for u, sigma in enumerate(S_i):
                val = np.trace(R @ sigma)
                val = float(np.real_if_close(val))
                if val < 0 and val > -1e-12:
                    val = 0.0
                P[r, u] = val

        # Pseudoinverse
        Phi = np.linalg.pinv(P)

        # Ensure Phi has same (n_R, n_S) shape for consistent indexing.
        # np.linalg.pinv returns (n_S, n_R) if P is (n_R, n_S) and rank issues appear.
        # So project back to shape (n_R, n_S) (this is a convention choice).
        if Phi.shape != P.shape:
            Phi = Phi.T
            if Phi.shape != P.shape:
                raise RuntimeError(f"Unexpected shape for Phi of party {i}: {Phi.shape}, P={P.shape}")

        P_matrices.append(P)
        Phi_matrices.append(Phi)
        n_R_per_party.append(n_R)
        n_S_per_party.append(n_S)

    # 2) Joint reference POVM elements and states
    #    R_s̃ = ⊗_i R_{s_i}, sigma_ũ = ⊗_i sigma_{u_i}
    ref_outcome_tuples = list(product(*[range(n_R) for n_R in n_R_per_party]))
    ref_state_tuples   = list(product(*[range(n_S) for n_S in n_S_per_party]))

    joint_reference_measurement = []
    for s_tuple in ref_outcome_tuples:
        local_Rs = [reference_measurements[i][s_tuple[i]] for i in range(n_parties)]
        joint_reference_measurement.append(kron(*local_Rs))

    joint_reference_states = []
    for u_tuple in ref_state_tuples:
        local_sigmas = [reference_states[i][u_tuple[i]] for i in range(n_parties)]
        joint_reference_states.append(kron(*local_sigmas))

    # 3) Joint Phi vector
    Phi_joint = kron(*Phi_matrices)  # multi-index: (r_0,s_0,r_1,s_1,...)
    phi = Phi_joint.flatten()

    meta = {
        "n_parties": n_parties,
        "n_R_per_party": n_R_per_party,
        "n_S_per_party": n_S_per_party,
        "ref_outcome_tuples": ref_outcome_tuples,
        "ref_state_tuples": ref_state_tuples,
    }
    return (phi, Phi_matrices, P_matrices, joint_reference_measurement, joint_reference_states, meta)

def construct_T(state,
                bell_povms,
                reference_measurements,
                reference_states,
                return_metadata=False):
    """
    Construct the T matrix such that p = T @ phi, where:

        - p is the Bell behavior vector you would get from quantum_behavior_from_povms(bell_povms, state),
        - phi is the vector from construct_joint_reference(reference_measurements, reference_states),
        - ref_meta is the metadata returned by construct_joint_reference (needed for indexing).

    Parameters
    ----------
    state : np.ndarray (D x D)
        Global density matrix on the tensor product Hilbert space.
    bell_povms : list
        bell_povms[party][measurement][outcome] are POVM elements.
    reference_measurements : list
        reference_measurements[party][r] are POVM elements R_r^(i).
    reference_states : list
        reference_states[party][r] are reference states sigma_r^(i).
    joint_reference_measurement : list of np.ndarray
        List of global POVM elements (R_{s_0} ⊗ ... ⊗ R_{s_{n-1}}),
        in the same order as ref_meta["ref_outcome_tuples"].
    ref_meta : dict
        Metadata from construct_joint_reference, with keys:
        - "n_parties"
        - "n_R_per_party"
        - "n_S_per_party"
        - "ref_outcome_tuples"
        - "ref_state_tuples"

    Returns
    -------
    T : np.ndarray
        Shape (num_bell_probs, num_phi_entries),
        such that p = T @ phi.
    meta : dict
        Contains:
        - "settings": list of tuples (x_0,...,x_{n-1})
        - "outcome_ranges": list of lists of outcome counts per setting
        - "phi_index_tuples": list of 2n-tuples (r_0,s_0,...,r_{n-1},s_{n-1})
    """
    phi, Phi_mats, P_mats, joint_reference_measurement, joint_reference_states, ref_meta = construct_joint_reference(reference_measurements, reference_states)

    n_parties = ref_meta["n_parties"]
    n_R_per_party = ref_meta["n_R_per_party"]
    ref_outcome_tuples = ref_meta["ref_outcome_tuples"]  # all s-tuples

    # --- 1) Compute joint reference distribution PR(s_0,...,s_{n-1}) from state ---
    PR = []
    for R_joint in joint_reference_measurement:
        val = np.trace(R_joint @ state)
        val = float(np.real_if_close(val))
        if val < 0 and val > -1e-12:
            val = 0.0
        PR.append(val)
    PR = np.array(PR, dtype=float)  # shape (#ref_outcome_tuples,)

    # Map s-tuple -> PR index
    s_index_map = {s_tuple: idx for idx, s_tuple in enumerate(ref_outcome_tuples)}

    # --- 2) Precompute single-party conditional probabilities P_i(a_i | x_i, R_{r_i}) ---
    cond_probs = []  # cond_probs[party][measurement] has shape (n_out, n_R_i)
    for i in range(n_parties):
        party_povms = bell_povms[i]
        R_i = reference_measurements[i]
        n_R = len(R_i)

        # infer local dimension
        d_i = R_i[0].shape[0]
        for R in R_i:
            if R.shape != (d_i, d_i):
                raise ValueError(f"All reference POVM elements for party {i} must have same shape.")
        for m, povm in enumerate(party_povms):
            for o, M in enumerate(povm):
                if M.shape != (d_i, d_i):
                    raise ValueError(f"POVM for party {i}, meas {m}, outcome {o} has wrong shape {M.shape}.")

        party_cond = []
        # Here we follow eq. (3.2)-style: P(a|A_x,R_r) = Tr(Π_{a|x} sigma_r) with sigma_r^i = reference_states[i][r].
        for m, povm in enumerate(party_povms):
            n_out = len(povm)
            mat = np.zeros((n_out, n_R), dtype=float)
            for r, sigma_r in enumerate(reference_states[i]):
                for a, M in enumerate(povm):
                    val = np.trace(M @ sigma_r)
                    val = float(np.real_if_close(val))
                    if val < 0 and val > -1e-12:
                        val = 0.0
                    mat[a, r] = val
            party_cond.append(mat)
        cond_probs.append(party_cond)

    # --- 3) Enumerate Bell settings & outcomes (same order as quantum_behavior_from_povms) ---
    settings_list = list(product(*[range(len(bell_povms[i])) for i in range(n_parties)]))

    outcome_ranges_per_setting = []
    num_bell_probs = 0
    for setting in settings_list:
        out_counts = [len(bell_povms[i][setting[i]]) for i in range(n_parties)]
        outcome_ranges_per_setting.append(out_counts)
        num_bell_probs += int(np.prod(out_counts))

    # --- 4) Phi index tuples: (r_0,s_0,...,r_{n-1},s_{n-1}) ---
    r_ranges = [range(n_R) for n_R in n_R_per_party]
    s_ranges = [range(n_R) for n_R in n_R_per_party]  # same number of columns as rows (n_R) per party
    phi_index_tuples = []
    for r_tuple in product(*r_ranges):
        for s_tuple in product(*s_ranges):
            phi_index_tuples.append((r_tuple, s_tuple))
    num_phi_entries = len(phi_index_tuples)

    # --- 5) Build T ---
    T = np.zeros((num_bell_probs, num_phi_entries), dtype=float)
    row = 0

    for s_idx, setting in enumerate(settings_list):
        out_counts = outcome_ranges_per_setting[s_idx]
        outcome_ranges = [range(c) for c in out_counts]

        for outcome_tuple in product(*outcome_ranges):
            # For each column of T, corresponding to (r_tuple, s_tuple)
            for col, (r_tuple, s_tuple) in enumerate(phi_index_tuples):
                # Product over parties of P_i(a_i | x_i, R_{r_i})
                prob_prod = 1.0
                for i in range(n_parties):
                    x_i = setting[i]
                    a_i = outcome_tuple[i]
                    r_i = r_tuple[i]
                    prob_prod *= cond_probs[i][x_i][a_i, r_i]

                # Joint reference probability PR(s_tuple)
                s_idx_PR = s_index_map[s_tuple]
                prob_prod *= PR[s_idx_PR]

                T[row, col] = prob_prod
            row += 1

    meta = {
        "settings": settings_list,
        "outcome_ranges": outcome_ranges_per_setting,
        "phi_index_tuples": phi_index_tuples,
        "phi": phi
    }

    return (T, meta) if return_metadata else T

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