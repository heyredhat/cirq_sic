import cirq 
import numpy as np

H = cirq.H
R = lambda k: cirq.ZPowGate(exponent=2**(1-k))
CR = lambda k: cirq.CZPowGate(exponent=2**(1-k))
SWAP = cirq.SWAP

# Qudit Fourier transform
def F(q):
    n = len(q)
    for i in range(n):
        yield H(q[i])
        for j in range(i + 1, n):
            yield CR(j - i + 1)(q[j], q[i])
    for i in range(n // 2):
        yield SWAP(q[i], q[n - 1 - i])

# Qudit inverse Fourier transform
def Fdag(q):
    yield cirq.inverse(list(F(q)))

# Qudit clock
def Z(q):
    for i in range(len(q)):
        yield R(i+1)(q[i])

# Qudit inverse clock
def Zdag(q):
     yield cirq.inverse(list(Z(q)))

# Qudit shift
def X(q):
    yield F(q)
    yield Z(q)
    yield Fdag(q)

# Qudit inverse shift
def Xdag(q):
     yield cirq.inverse(list(X(q)))

# Qubit controlled clock
def QCZ(c, t):
    n = len(t)
    for j in range(n):
        yield CR(j+1)(c, t[j])

# Qudit controlled clock
def CZ(c, t):
    n = len(c)
    for j in range(n):
        for k in range(2**j):
            yield QCZ(c[n-j-1], t)

# Qudit inverse controlled clock
def CZdag(c, t):
    yield cirq.inverse(list(CZ(c, t)))

# Qudit controlled shift
def CX(c, t):
    yield F(t)
    yield CZ(c, t)
    yield Fdag(t)

# Qudit inverse controlled clock
def CXdag(c, t):
    yield cirq.inverse(list(CX(c, t)))

# Qudit Arthurs-Kelly 
def AK(c, t1, t2):
    yield CXdag(c, t1)
    yield Fdag(c)
    yield CXdag(c, t2)
    yield F(c)

# Prepare Arthurs-Kelly ancillas 
def AP(t1, t2):
    yield Fdag(t2)
    yield CZ(t1, t2)

Ry = lambda t: cirq.Ry(rads=t)
Sx = cirq.X
H = cirq.H
Ph = lambda t: cirq.ZPowGate(exponent=t/np.pi)
CNOT = cirq.CNOT

# Controlled y-rotation
def CRy(theta):
	def __CRy__(c, t):
		yield Ry(theta/2)(t)
		yield CNOT(c, t)
		yield Ry(-theta/2)(t)
		yield CNOT(c, t)
	return __CRy__

# Prepare an almost flat d=4 monomial fiducial
def monomial_fiducial(q):
    theta1 = 2*np.arccos(np.sqrt((5+np.sqrt(5))/10))
    theta2 = 2*np.arccos(np.sqrt(1+np.sqrt(5))/2)
    theta3 = np.pi/2

    yield Ry(theta1)(q[0])
    yield Sx(q[0])
    yield CRy(theta2)(q[0], q[1])
    yield Sx(q[0])
    yield CRy(theta3)(q[0], q[1])

# Rephase the d=4 monomial basis
def monomial_rephasing(q):
    yield Ph(np.pi)(q[1])
    yield CNOT(q[0], q[1])
    yield Ph(3*np.pi/4)(q[1])
    yield CNOT(q[0], q[1])
    yield Ph(-np.pi/2)(q[0])

# Prepare a d=4 SIC fiducial (or its conjugate)
def d4_fiducial(q, conjugate=False):
	yield monomial_fiducial(q)
	yield monomial_rephasing(q) if not conjugate else \
		  cirq.inverse(list(monomial_rephasing(q)))
	yield H(q[0])