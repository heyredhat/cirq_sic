import numpy as np
import scipy as sc

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)

from .sic import *

def frame_potential_minimum(d, t):
    return 1/sc.special.binom(d+t-1, t)

def minimize_wh_frame_potential(d, T=10):
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
                                    tol=1e-32, method="SLSQP", options={"disp": False, "maxiter": 10000})
                                        for t in range(T)]
    V = results[np.argmin([r.fun for r in results])].x
    ket = jp.concatenate([jp.array([1]), V[:d-1] + 1j*V[d-1:]])
    return np.array(ket/jp.linalg.norm(ket))