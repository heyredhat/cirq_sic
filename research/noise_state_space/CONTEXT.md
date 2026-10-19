# Noise reconstruction study — 2026-10-18

User objective: systematically compare state spaces inferred from noisy real SIC
triples with true geometry and with random-state SIC tomography; assess hardware
diagnostics and limitations; promote validated tools into the library.

Design: independent circuit-level density-matrix simulations; exact channel
expectations separated from finite shots. Start d=2 full ordered triples, expand
to d=4 symmetry-imposed exploratory data. Use nominal ideal SIC coordinates and
ideal Gram throughout; explicitly do not claim fitted physical SICs. Preserve raw
ordered settings to test symmetry before averaging. Compare radial geometry only
when uniform center is strictly feasible. Tomography clouds retain unphysical
linear inversions, with no positivity clipping. Preparation, per-moment, and
readout errors are distinct experiments. Common coherent rotations are an
essential counterexample to diagnostic identifiability.

Runtime: /home/heyredhat/anaconda3/envs/cirq_sic/bin/python (Cirq 1.6.1).
Prototype: pipeline.py; promote after analytic/circuit cross-checks.
References: https://arxiv.org/abs/0910.2750 ; https://arxiv.org/abs/1001.0004 ;
https://quantumai.google/cirq/simulate/noisy_simulation .

## Completed implementation and validation

Promoted the validated prototype into `cirq_sic/noise.py`, with seven lazy public
exports. The original `pipeline.py` is archived provenance, not maintained API.
`run_study.py` now imports only the library implementation. Added full docs in
`docs/noise_state_space.md` and links from API/root/research documentation.

All 283 tests passed (`python -m pytest -q`, 19.37 s, recorded below).
New independent Kraus checks caught a prototype placement error: the control H
shared the preparation moment, so insertion before H initially preceded state
preparation. Corrected by splitting that moment; regenerated all study results.
Depolarization alone had masked the error through unitary covariance. All five
noise families plus a coherent rotation now match independent predictions.
The regression suite also exposed an existing normalization idempotency problem
in OutcomeData JSON round trips. A minimal machine-precision tolerance and a
regression test now preserve saved probabilities exactly.

## Saved experiments

- 39 d=2 scenarios, 64 ordered real triple circuits each, 4096 shots/setting.
- 384 shared Haar states for each matched tomography scenario; 4096 shots/state.
- Exact and sampled arrays, settings, counts, tensors, JSON summaries retained.
- 100 bootstrap tensors evaluated for selected cases, failed centers counted.
- d=4 ideal, readout .01, preparation depolarization .01: 816 unordered settings
  each; symmetry is imposed, so these cannot diagnose register order faults.
- CZ compilation supplement: mean 43 moments versus 4 logical moments; per-moment
  depolarization .001 gives volume ratio .36942; .01 certifies an empty body.
- Equal-total-shot supplement: tomography 682 shots/state (261888 total) versus
  triple 262144; fresh counts and metrics saved separately.
- `analyze_results.py` refreshes affine-fit diagnostics from saved clouds and
  creates figures/table. `supplementary.py` reproduces extra comparisons.

## Main conclusions and quantitative anchors

Nominal ideal-Gram reconstruction and noisy tomography estimate different bodies.
Preparation depolarization with lambda=1-4p/3 gives triple Bloch radius
sqrt(3 lambda^2-2)/lambda^2, versus tomography lambda. At p=.05 the volumes are
.726648 and .813037. At p=.15 the triple body is analytically empty, while the
physical tomography image retains .512 volume. Readout attenuation s=1-2e
contracts the triple body by ((d+1)s-d)/s; volume raises this to d^2-1.
A common Rz(.4) is invisible to triples and shape, but paired tomography TV=.08006.
Do not infer a unique hardware channel from a distortion signature.
At readout e=.15, 13/100 bootstrap centers fail; at e=.2, 100/100 fail.
The ideal 384-state convex hull covers .968963 of the true ball volume.
d=4 depolarizing preparation volume .14034 has Monte Carlo SE .00542 and is
exploratory; no d=4 tomography sweep was run.

## Report and reproducibility status

Seven-page `report/main.pdf` and editable `report/main.tex` compiled successfully
in both local TeX and the desktop editor. All seven rendered pages visually
reviewed; no overflow warnings. Figures include preparation-noise volume curves
and two-coordinate projections (explicitly not sections). References are in TeX.
The report distinguishes exact analytic claims, sampled metrics, model-dependent
affine extrapolation, unimplemented features, and a proposed hardware protocol.

Next meaningful extensions: calibrated native timings/routing; independent SPAM
and ancilla controls; ordered d=4 experiments; general SDP feasibility/recentering;
high-dimensional convergence and repeated-experiment interval coverage studies.
These are research extensions, not unreported completed work.
