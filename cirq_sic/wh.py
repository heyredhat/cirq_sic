import numpy as np
from numpy.linalg import matrix_power as mpow

from .utils import *

####################################################################################

def wh_operators(d):
    """Construct WH operators."""
    w = np.exp(2*np.pi*1j/d)
    Z = np.diag(np.array([w**i for i in range(d)]))
    F = np.array([[w**(i*j) for j in range(d)] for i in range(d)])/np.sqrt(d)
    X = F.conj().T @ Z @ F
    D = np.array([[mpow(X, i) @ mpow(Z, j)  for j in range(d)] for i in range(d)])
    return locals()

def wh_povm(phi):
	"""Generate WH-POVM elements from a fiducial ket"""
	d = phi.shape[0]
	D = wh_operators(d)["D"]
	Pi = np.outer(phi, phi.conj())
	return np.array([D[a].conj().T @ Pi @ D[a] for a in np.ndindex(d,d)])/d

def change_conjugate_convention(p):
    r"""If we have WH-POVM probabilities (a d^2 vector), reorder them from the convention $D^\dag \Pi D to D \Pi D^\dag (and vice versa)."""
    if len(p.shape) > 1:
         return np.array([change_conjugate_convention(p_i) for p_i in p]) 
    d = int(np.sqrt(p.shape[0]))
    idx_order = [0] + list(range(1, d))[::-1]
    return p.reshape(d,d)[np.ix_(idx_order, idx_order)].flatten()

def arthurs_kelly_ancilla_ready_state(phi, fourier=False):
    """From a WH fiducial state, calculate the proper ready state for the AK ancillas."""
    d = phi.shape[0]
    WH = wh_operators(d)
    F, w = WH["F"], WH["w"]
    Pi = np.outer(phi, phi.conj())
    FPi = F.conj().T @ Pi
    gamma = np.array([w**(k*m)*FPi[m,k] for k in range(d) for m in range(d)])
    return kron(F.conj().T, F.conj().T) @ gamma if fourier else gamma
