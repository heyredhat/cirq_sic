# Noise in SIC state-space reconstruction

This is a reproducible research study and its continuation record, started
2026-10-18. The maintained implementation is [`cirq_sic.noise`](../../cirq_sic/noise.py).
The original prototype is retained as `pipeline.py` for provenance; do not import
it in new analyses. The running LaTeX project is [`report/main.tex`](report/main.tex).

Run from the repository root:

```sh
python research/noise_state_space/run_study.py
python research/noise_state_space/analyze_results.py
python research/noise_state_space/supplementary.py
python -m pytest tests/test_noise.py
cd research/noise_state_space/report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For LaTeX, compiling with working directory `report/` is preferred because figures
are referenced relatively. The built-in editor also supports this project.
`run_study.py --quick` is a smoke run and **overwrites the same result paths**;
use a separate checkout/output copy to preserve the full study.

- `results/manifest.json`: seed, versions, shot/state counts and elapsed time.
- `results/summary.json`: metrics for all qubit cases, including failures.
- `results/d4_summary.json`: exploratory d=4 checks (symmetry imposed).
- `results/*.npz`: states, settings, counts, expected and sampled probabilities.
- `analyze_results.py`: refresh derived metrics, figures and report tables.
- `CONTEXT.md`: decisions, validation history and next steps.

All findings concern specified logical circuits and memoryless simulated channels.
No hardware data were acquired. The report distinguishes established analytic
facts from numerical results and a proposed device-validation protocol.
