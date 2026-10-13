"""Check installation-facing API stability and absence of import side effects."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
import cirq_sic


def test_package_import_is_lazy_and_does_not_change_numpy_options():
    code = """
import sys
import numpy as np
before = np.get_printoptions().copy()
import cirq_sic
assert before == np.get_printoptions()
assert 'cirq' not in sys.modules
for name in ('jax', 'recirq', 'qiskit', 'cvxpy', 'bqskit', 'cirq_google', 'qsimcirq', 'plotly'):
    assert name not in sys.modules, name
assert cirq_sic.__version__ == '0.2.0'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_declared_exports_resolve_without_optional_backends():
    for name in cirq_sic.__all__:
        value = getattr(cirq_sic, name)
        if callable(value):
            assert value.__doc__, name
    assert "paulis" in cirq_sic.__all__
    assert "bell_overlap_circuit" in cirq_sic.__all__
    assert "wh_entropy_from_overlaps" in cirq_sic.__all__
    assert not {"np", "cirq", "sc", "mpow", "jax"} & set(cirq_sic.__all__)
    assert set(cirq_sic.__all__).issubset(dir(cirq_sic))
    with pytest.raises(AttributeError):
        getattr(cirq_sic, "not_a_library_function")


def test_bundled_fiducial_dimensions_are_documented():
    from cirq_sic.sics import load_sic_fiducial

    paths = list((Path(cirq_sic.__file__).parent / "sic_povms").glob("d*.txt"))
    dimensions = {int(path.stem[1:]) for path in paths}
    assert dimensions == set(range(2, 152)) - {72}
    # Read the largest resource to catch wheel packaging/path regressions,
    # without allocating its d^4 displacement matrices.
    assert load_sic_fiducial(151).shape == (151,)


def test_all_runtime_modules_have_module_documentation():
    root = Path(cirq_sic.__file__).parent
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        assert ast.get_docstring(ast.parse(path.read_text())), str(path)
