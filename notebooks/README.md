# Maintained demonstrations

Install with `python -m pip install -e '.[notebooks,test]'` from the repository
root. Open **[Tutorial.ipynb](Tutorial.ipynb)** and use the same environment's
Python kernel.

| Notebook | What it demonstrates |
| --- | --- |
| [Tutorial](Tutorial.ipynb) | All four workflows, conventions, plots, artifacts, and hardware gate synthesis |
| [EmbeddingTest](EmbeddingTest.ipynb) | Full Born distributions and zero padding in dimensions 2–8; mixed qutrits |
| [TestingMixedSE](TestingMixedSE.ipynb) | Direct two-copy WH Rényi-2 entropy, purity, mixed-state caveats |
| [SICEntropy](SICEntropy.ipynb) | Distinction between fixed-SIC outcome entropy and stabilizer entropy |
| [TaskExample](TaskExample.ipynb) | Reusable plans, JSON artifacts, calibrated sky/ground comparisons |
| [TripleProducts](TripleProducts.ipynb) | Real/imaginary measurements, tensor symmetries, positivity and state-space sections |

All notebooks use seeded randomness, modest dimensions and temporary data.
They require no hardware account or pre-existing measurement files.

Run `python tools/execute_notebooks.py` from the repository to execute each in a
fresh kernel. Outputs go to `artifacts/notebooks`; `--write` updates the source
notebooks. The runner uses its own Python interpreter for the kernels.

The [research archive](../research/README.md) preserves original experiments
and outputs, including the earlier embedding attempts. Those notebooks are not
part of this maintained suite.
