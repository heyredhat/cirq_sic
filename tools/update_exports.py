"""Regenerate the explicit lazy public-export map after intentional API changes."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "ansatz",
    "circuits",
    "sics",
    "utils",
    "wh",
    "geometry",
    "triples",
    "experiments.experiment_utils",
    "experiments.sky_ground",
    "experiments.sky_ground_analysis",
    "measurements",
    "analysis",
    "workflow",
]
exports = {}
for module in MODULES:
    path = ROOT / "cirq_sic" / (module.replace(".", "/") + ".py")
    if not path.exists():
        continue
    tree = ast.parse(path.read_text())
    public_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                public_names = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    if public_names is None:
        public_names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
    for name in public_names:
        exports[name] = module
header = '''"""SIC/WH measurements and reproducible quantum experiments.

Public objects are imported lazily. Importing cirq_sic does not configure NumPy,
initialize JAX, connect hardware, or import plotting/compiler integrations.
Prefer explicit imports from the focused modules for discoverable dependencies.
"""
from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
'''
lines = [f"    {name!r}: {module!r}," for name, module in sorted(exports.items())]
footer = '''
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Resolve a declared public object on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public exports in interactive completion."""
    return sorted(set(globals()) | set(__all__))
'''
(ROOT / "cirq_sic/__init__.py").write_text(header + "\n".join(lines) + footer)
