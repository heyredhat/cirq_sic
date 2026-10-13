"""Full-unitary regression tests for padded and degenerate synthesis inputs."""
import cirq
import numpy as np
import pytest
from scipy.stats import unitary_group

from cirq_sic._synthesis import shannon_operations
from cirq_sic.workflow import compile_circuit


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("kind", ["identity", "diagonal", "random"])
def test_synthesis_preserves_entire_unitary(n, kind):
    q = cirq.LineQubit.range(n)
    if kind == "identity":
        matrix = np.eye(2**n)
    elif kind == "diagonal":
        matrix = np.diag(np.exp(1j * np.repeat([0.3, 0.7], 2 ** (n - 1))))
    else:
        matrix = unitary_group.rvs(2**n, random_state=47)
    decomposed = cirq.Circuit(shannon_operations(q, matrix))
    assert np.allclose(decomposed.unitary(qubit_order=q), matrix, atol=1e-10)
    compiled = compile_circuit(cirq.Circuit(cirq.MatrixGate(matrix).on(*q)))
    assert all(len(op.qubits) <= 2 for op in compiled.all_operations())
    # CZ synthesis may discard an unobservable overall phase.
    cirq.testing.assert_allclose_up_to_global_phase(
        compiled.unitary(qubit_order=q), matrix, atol=1e-7
    )
