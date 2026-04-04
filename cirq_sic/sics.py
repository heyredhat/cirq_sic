from importlib.resources import files

import numpy as np
import scipy as sc

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)

from .wh import *

####################################################################################

def load_sic_fiducial(d):
    r"""
    Loads a Weyl-Heisenberg covariant SIC-POVM fiducial state of dimension $d$ from the repository provided here:
      http://www.physics.umb.edu/Research/QBism/solutions.html.
    """
    data_file_path = files("cirq_sic").joinpath("sic_povms/d%d.txt" % d)
    with data_file_path.open("r") as f:
        fiducial = []
        for line in f:
            if line.strip() != "":
                re, im = [float(v) for v in line.split()]
                fiducial.append(re + 1j*im)
        fiducial = np.array(fiducial)
        return fiducial/np.linalg.norm(fiducial)

def SIC_P(d):
    """Conditional probability matrix: SIC outcomes given SIC states."""
    return np.array([[(d*(1 if i == j else 0) + 1)/(d*(d+1)) for j in range(d**2)] for i in range(d**2)])

####################################################################################

def frame_potential_minimum(d, t=2):
    """Minimum value of the frame potential. SICs minimize for t=2."""
    return 1/sc.special.binom(d+t-1, t)

def minimum_design_elements(d, t):
    """Minimum number of elements for a t-design."""
    return sc.special.binom(d-1+np.floor(t/2), np.floor(t/2))*\
           sc.special.binom(d-1+np.ceil(t/2), np.ceil(t/2))

######################
def minimize_wh_frame_potential(d, T=10, method="SLSQP"):
    """Find a SIC fiducial by minimizing the frame potential under assumption of WH covariance."""
    D = wh_operators(d)["D"]
    fp_min = frame_potential_minimum(d, 2)

    @jax.jit
    def wh_frame_potential(V):
        ket = jp.concatenate([jp.array([1]), V[:d-1] + 1j*V[d-1:]])
        ket = ket/jp.linalg.norm(ket)
        R = jp.einsum("ijkl, l->ijk", D, ket).reshape(d**2, d)
        return (jp.sum(abs(R @ R.conj().T)**4)/d**4 - fp_min)**2

    results = [sc.optimize.minimize(wh_frame_potential, np.random.randn(2*d-2),\
                                    jac=jax.jit(jax.jacrev(wh_frame_potential)),\
                                    tol=1e-32, method=method, options={"disp": False, "maxiter": 10000})
                                        for t in range(T)]
    V = results[np.argmin([r.fun for r in results])].x
    ket = jp.concatenate([jp.array([1]), V[:d-1] + 1j*V[d-1:]])
    return np.array(ket/jp.linalg.norm(ket))

def wh_2_frame_potential(ket):
    """Returns the 2-frame potential of a ket. Minimized for a SIC."""
    d = ket.shape[0]
    D = wh_operators(d)["D"]
    R = jp.einsum("ijkl, l->ijk", D, ket).reshape(d**2, d)
    return (jp.sum(abs(R @ R.conj().T)**4)/d**4 - frame_potential_minimum(d, 2))**2

def frame_potential(R, t):
    """t-th order frame potential for n vectors given as a d x n matrix."""
    norms = np.linalg.norm(R, axis=0)
    R = R/norms
    p = norms/np.sum(norms)
    P = jp.outer(p, p)
    return np.sum(P*abs(R.conj().T @ R)**(2*t))

##############################################################

def d4_sic_fiducial_ket(monomial=False):
    """Construct a d=4 SIC fiducial ket from the monomial basis."""
    monomial_fiducial = (np.array([np.sqrt(2 + np.sqrt(5)), 1, 1, 1])/np.sqrt(5 + np.sqrt(5)))
    if monomial:
        return monomial_fiducial 
    H = np.array([[1,1],[1,-1]])/np.sqrt(2)
    P = np.diag([1, np.exp(1j*np.pi*(-1/4)), np.exp(1j*np.pi*(1/4)), np.exp(1j*np.pi*(1/2))]) 
    return np.kron(H, np.eye(2)) @ P @ monomial_fiducial

####################################################################################

def all_d2_fiducials():
    """Returns all dimension 2 SIC states, which form two antipodal tetrahedra in the Bloch sphere."""
    D = wh_operators(2)["D"].reshape(-1, 2, 2)
    phi = np.array([np.sqrt(3 + np.sqrt(3)), np.exp(1j*np.pi/4)*np.sqrt(3 - np.sqrt(3))])/np.sqrt(6)
    phi_antipodal = np.array([np.exp(-1j*np.pi/4)*np.sqrt(3 - np.sqrt(3)), np.sqrt(3 + np.sqrt(3))])/np.sqrt(6)
    states = [O @ phi for O in D] + [O @ phi_antipodal for O in D]
    return np.array(states)
