import numpy as np
from itertools import product
from functools import reduce
import scipy as sc

from .utils import *

def grey(n):
    L = [[0],[1]]
    while len(L) < 2**n:
        L = [[0]+_ for _ in L] + [[1]+_ for _ in L[::-1]]
    return np.array(L)

def grey_data(k):
    if k == 0:
        return None
    B, G = np.array(list(product([0,1], repeat=k))), grey(k)
    targets = [np.where(G[i] != G[(i+1) % 2**k])[0][0] for i in range(2**k)]
    M = 2**(-k) * np.array([[(-1)**(B[j] @ G[i]) for j in range(2**k)] for i in range(2**k)])
    return M, targets

def ansatz_params_to_angles(n, params, sign=1):
    """Converts a parameter vector back into ansatz angles."""
    phase = params[0]
    params = params[1:]
    m = int(len(params)/2)
    r = [2**i for i in range(n-1)]
    thetas = np.split(params[:m], [np.sum(r[:i]) for i in range(1,n)])
    phis = np.split(sign*params[m:], [np.sum(r[:i]) for i in range(1,n)])
    return thetas, phis, phase

def ansatz_angles_to_params(thetas, phis, phase):
    """Flattens angle matrices into a parameter vector."""
    return np.concatenate([[phase], np.concatenate(thetas), np.concatenate(phis)])

def ket_to_ansatz_angles(ket):
    """Converts a ket to the ansatz angles."""
    kets, thetas, phis = [ket], [], []
    for j in range(int(np.log2(len(ket)))):
        current_thetas, current_phis, current_zs = [], [], []
        for i in range(0, len(kets[-1]), 2):
            qubit = kets[-1][i:i+2]
            normalized_qubit = np.exp(-1j*np.angle(qubit[0]))*qubit/np.linalg.norm(qubit)
            theta = float(2*np.arccos(normalized_qubit[0]).real)
            phi = float(np.angle(normalized_qubit[1]))
            U = sc.linalg.expm(-1j*sigma_z*phi/2) @ sc.linalg.expm(-1j*sigma_y*theta/2)
            z = (U.conj().T @ qubit)[0]
            current_thetas.append(theta)
            current_phis.append(phi)
            current_zs.append(z)
        kets.append(np.array(current_zs))
        thetas.append(np.array(current_thetas))
        phis.append(np.array(current_phis))
    phase = np.angle(kets[-1][0])
    return thetas[::-1], phis[::-1], phase

def ansatz_unitary(n, params):
    """Constructs the unitary corresponding to the ansatz for n qubits and a vector of parameters."""
    thetas, phis, phase = ansatz_params_to_angles(n, params, sign=1)
    T = thetas[::-1]
    P = phis[::-1]
    return reduce(np.matmul, [np.kron(sc.linalg.block_diag(*\
                                        [sc.linalg.expm(-1j*sigma_z*P[i][j]/2) @ \
                                         sc.linalg.expm(-1j*sigma_y*T[i][j]/2)\
                                                for j in range(len(T[i]))]),\
                                         np.eye(2**i)) for i in range(len(T))]+[np.kron(sc.linalg.expm(1j*sigma_z*phase), np.eye(2**(n-1)))])

def __ansatz_circuit__(q, params, conjugate=False):
    n = len(q)
    targeting_data = [grey_data(i) for i in range(n)]
    sign = -1 if conjugate else 1
    thetas, phis, phase = ansatz_params_to_angles(n, params, sign=sign)
    
    yield cirq.Rz(rads=-2*sign*phase)(q[0])
    yield cirq.Ry(rads=thetas[0][0])(q[0])
    yield cirq.Rz(rads=phis[0][0])(q[0])
    for i in range(1, len(thetas)):
        current_q = q[:i+1]
        M, targets = targeting_data[i]
        for j, theta in enumerate(M @ thetas[i]):
            yield cirq.Ry(rads=theta)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])
        for j, phi in enumerate(M @ phis[i]):
            yield cirq.Rz(rads=phi)(current_q[-1])
            yield cirq.CNOT(current_q[targets[j]], current_q[-1])

def ansatz_circuit(ket):
    params = ansatz_angles_to_params(*ket_to_ansatz_angles(ket))
    def __ansatz__(q, conjugate=False):
        yield __ansatz_circuit__(q, params, conjugate=conjugate)
    return __ansatz__