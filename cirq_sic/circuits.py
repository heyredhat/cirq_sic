import cirq 
import numpy as np

H = cirq.H
R = lambda k: cirq.ZPowGate(exponent=2**(1-k))
CR = lambda k: cirq.CZPowGate(exponent=2**(1-k))
SWAP = cirq.SWAP

def F(q):
    """Qudit Fourier transform"""
    n = len(q)
    for i in range(n):
        yield H(q[i])
        for j in range(i + 1, n):
            yield CR(j - i + 1)(q[j], q[i])
    for i in range(n // 2):
        yield SWAP(q[i], q[n - 1 - i])

def Fdag(q):
    """Qudit inverse Fourier transform"""
    yield cirq.inverse(list(F(q)))

def Z(q):
    """Qudit clock"""
    for i in range(len(q)):
        yield R(i+1)(q[i])

def Zdag(q):
     """Qudit inverse clock"""
     yield cirq.inverse(list(Z(q)))

def X(q):
    """Qudit shift"""
    yield F(q)
    yield Z(q)
    yield Fdag(q)

def Xdag(q):
     """Qudit inverse shift"""
     yield cirq.inverse(list(X(q)))

def displace(q, a1, a2):
    """Act with WH operator with indices (a1, a2)."""
    for i in range(a2):
        yield Z(q)
    for j in range(a1):
        yield X(q)

def wh_state(q, a1, a2, fiducial_prep):
    """Prepare the WH state D(a1,a2)|fiducial>"""
    yield fiducial_prep(q)
    yield displace(q, a1, a2)

def QCZ(c, t):
    """Qubit controlled clock"""
    n = len(t)
    for j in range(n):
        yield CR(j+1)(c, t[j])

def CZ(c, t):
    """Qudit controlled clock"""
    n = len(c)
    for j in range(n):
        for k in range(2**j):
            yield QCZ(c[n-j-1], t)

def CZdag(c, t):
    """Qudit inverse controlled clock"""
    yield cirq.inverse(list(CZ(c, t)))

def CX(c, t):
    """Qudit controlled shift"""
    yield F(t)
    yield CZ(c, t)
    yield Fdag(t)

def CXdag(c, t):
    """Qudit inverse controlled shift"""
    yield cirq.inverse(list(CX(c, t)))

def AP(t1, t2):
    """Prepare Arthurs-Kelly ancillas"""
    yield Fdag(t2)
    yield CZ(t1, t2)

def AK(c, t1, t2, measure=True):
    """Qudit Arthurs-Kelly """
    yield CXdag(c, t1)
    yield Fdag(c)
    yield CXdag(c, t2)
    yield F(c)
    if measure:
        yield cirq.measure(*[t1+t2], key="result")

def simple_AK(q, conj_fiducial, measure=True):
    """Simple Arthurs-Kelly"""
    yield CXdag(conj_fiducial, q)
    yield Fdag(conj_fiducial)
    if measure:
        yield cirq.measure(*(q+conj_fiducial), key="result")

Ry = lambda t: cirq.Ry(rads=t)
Sx = cirq.X
H = cirq.H
Ph = lambda t: cirq.ZPowGate(exponent=t/np.pi)
CNOT = cirq.CNOT

def CRy(theta):
	"""Controlled y-rotation"""
	def __CRy__(c, t):
		yield Ry(theta/2)(t)
		yield CNOT(c, t)
		yield Ry(-theta/2)(t)
		yield CNOT(c, t)
	return __CRy__

def d4_sic_monomial_fiducial(q):
    """Prepare an almost flat d=4 monomial SIC fiducial"""
    theta1 = 2*np.arccos(np.sqrt((5+np.sqrt(5))/10))
    theta2 = 2*np.arccos(np.sqrt(1+np.sqrt(5))/2)
    theta3 = np.pi/2

    yield Ry(theta1)(q[0])
    yield Sx(q[0])
    yield CRy(theta2)(q[0], q[1])
    yield Sx(q[0])
    yield CRy(theta3)(q[0], q[1])

def d4_monomial_rephasing(q):
    """Rephase the d=4 monomial basis"""
    yield Ph(np.pi)(q[1])
    yield CNOT(q[0], q[1])
    yield Ph(3*np.pi/4)(q[1])
    yield CNOT(q[0], q[1])
    yield Ph(-np.pi/2)(q[0])

def d4_sic_fiducial(q, conjugate=False):
	"""Prepare a d=4 SIC fiducial (or its conjugate)"""
	yield d4_sic_monomial_fiducial(q)
	yield d4_monomial_rephasing(q) if not conjugate else \
		  cirq.inverse(list(d4_monomial_rephasing(q)))
	yield H(q[0])
     
def qudit_basis_state(q, m):
    """Prepares the qudit basis state |m> on the qubits q."""
    n = len(q)
    bitstr = bin(m)[2:].zfill(n)
    for i, b in enumerate(bitstr):
        if b == '1':
            yield Sx(q[i])