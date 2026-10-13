# Mathematical conventions and estimator limits

This reference connects the four experiment families to the arrays and circuits
in `cirq_sic`. The [API guide](api.md) provides runnable examples. Equations
below describe ideal devices unless a finite-shot estimator is explicitly
introduced. A successful consistency check is evidence about the measured
probabilities and the assumed model; it does not uniquely identify hardware
errors or certify an entire device.

## Arrays, dimensions, and operator conventions

A ket is a normalized column vector in a logical Hilbert space of dimension
`d`. A density matrix satisfies `rho = rho†`, `rho >= 0`, and `Tr(rho) = 1`.
State arrays have shapes `(d,)` or `(d,d)`. A probability batch has shape
`(..., outcomes)`: the outcome axis is always last.

A rank-one POVM is represented by a weighted frame

\[
F=[f_0\;f_1\;\cdots\;f_{m-1}],\qquad
E_k=f_k f_k^\dagger,\qquad FF^\dagger=I_d.
\]

`Measurement.frame` has shape `(d,m)` and `Measurement.effects` has shape
`(m,d,d)`. Born probabilities are `p_k = Tr(rho E_k)`. Unweighted unit kets
are not a complete POVM frame unless their weights already equal one.
`Measurement.states` divides each frame column by its norm for use in
reference-state preparations.

Cyclic Weyl–Heisenberg operators use

\[
\omega=e^{2\pi i/d},\quad X|j\rangle=|j+1\pmod d\rangle,\quad
Z|j\rangle=\omega^j|j\rangle,\quad D_{a,b}=X^a Z^b.
\]

Displacements carry no extra phase. The flattened label is `k = a*d+b`,
with `(a,b)=(0,0)` first. Changing to `D† Pi D` sends both labels to their
negatives modulo `d`; `change_conjugate_convention` implements that permutation.

For a unit fiducial `phi`, the WH POVM is

\[
E_{a,b}=\frac1d D_{a,b}|\phi\rangle\langle\phi|D_{a,b}^\dagger.
\]

Every such orbit is a complete POVM, but not every fiducial produces an
informationally complete measurement. Rank is checked before tomography.
A WH SIC additionally satisfies the SIC overlaps given below.

The **cyclic WH basis at dimension four** and the **two-qubit tensor Pauli
basis** are different operator bases. The former uses `X_4` and `Z_4`; the
latter uses tensor products of `I,X,Y,Z`. Embedding the cyclic qudit into
qubits does not change which characteristic function an experiment measures.
This distinction is explicit in the entropy API's `basis` argument.

## Exact POVM embedding in any logical dimension

For `d >= 2`, the system occupies the first `d` basis states of
`n = ceil(log2(d))` qubits. A POVM with `m` rank-one outcomes uses
`N = ceil(log2(m))` measured qubits in total and `a=N-n` zero ancillas.
The implementation allocates at least one qubit for the trivial
one-dimensional system too.

Let `M=2**N`. Pad the analysis map to an isometry

\[
V=\begin{pmatrix}F^\dagger\\0\end{pmatrix}\in\mathbb C^{M\times d},
\qquad V^\dagger V=I_d.
\]

Registers use Cirq's big-endian convention, ordered **system then ancilla**.
The input `|j> |0...0>` therefore has integer index `j * 2**a`, not `j`.
The completed unitary is constrained by

\[
U_{:,j2^a}=V_{:,j},\qquad j=0,\ldots,d-1.
\]

The remaining columns form an orthonormal complement. This placement is
crucial when the ancilla is appended to the system. It gives

\[
\langle k|U(|\psi\rangle\otimes|0\cdots0\rangle)
=(F^\dagger\psi)_k,
\qquad k<m,
\]

so measuring the full register implements the desired POVM exactly in
floating-point arithmetic. The density-matrix result follows by linearity.
The physical output labels are the **contiguous integers `0..m-1`**; labels
`m..M-1` are padding. Input labels and output labels serve different roles.

`OutcomeData` retains all `M` probabilities. Its `valid` property is the
unnormalized physical portion and its `leakage` is the padding mass. Analysis
rejects appreciable padding rather than silently conditioning on success.
The tolerance is intended for roundoff, not for discarding physical leakage.
These padding outputs do not detect every possible error: errors can also
change probabilities among valid outputs.

The construction needs a valid supplied frame; it does not prove SIC existence
or discover a SIC in every dimension. `sic_povm` validates the supplied or
bundled fiducial. General rank-one frames use the same embedding independently
of whether a SIC is known.

A density matrix is prepared through the purification

\[
|\Psi_\rho\rangle=\sum_i\sqrt{\lambda_i}|v_i\rangle_S|i\rangle_E.
\]

The environment is separate from measurement ancillas and remains unmeasured.
This is a circuit for a specified preparation, not a procedure that knows an
unknown laboratory state. A POVM fixes probabilities but does not determine a
unique post-measurement instrument. The default measurement is destructive;
sky/ground reference preparations are separate explicit circuits.

## SIC tomography

Unscaled SIC projectors `Pi_i` obey

\[
\operatorname{Tr}\Pi_i=1,\quad
\sum_i\Pi_i=dI,\quad
G_{ij}=\operatorname{Tr}(\Pi_i\Pi_j)
=\frac{d\delta_{ij}+1}{d+1}.
\]

Their effects are `E_i=Pi_i/d`. Thus

\[
p_i=\frac1d\operatorname{Tr}(\rho\Pi_i),\qquad
\rho=(d+1)\sum_i p_i\Pi_i-I
=\sum_i\left((d+1)p_i-\frac1d\right)\Pi_i.
\]

`sic_reconstruction` checks these overlap conditions and applies the analytic
inverse. `linear_inversion` instead supports any informationally complete
POVM. It expands `rho=I/d+sum_l x_l H_l` in an orthonormal traceless Hermitian
basis, solves real least squares for `x`, and rejects deficient rank. The
result is Hermitian and trace one, but finite counts can yield negative
eigenvalues. With an exact fixed measurement model, this linear estimator is
unbiased for independently sampled frequencies.

`physical=True` projects the Hermitian estimate onto positive trace-one
matrices in Frobenius norm. Its eigenvalues are projected onto the probability
simplex: `lambda'_i=max(lambda_i-tau,0)` with `sum lambda'=1`. This differs from
clipping and renormalizing eigenvalues. Projection introduces bias and is not
maximum-likelihood tomography. Calibration uncertainty, model mismatch, and
readout errors are not absorbed into this projection.

The SIC probability representation and its role in the Born rule are developed
by [Fuchs and Schack, *Quantum-Bayesian Coherence*](https://arxiv.org/abs/0906.2187).

## Stabilizer Rényi entropy and mixed-state diagnostics

Choose a trace-orthogonal unitary basis `U`, with `Tr(U†V)=d delta(U,V)`.
`basis='pauli'` uses tensor Pauli operators and requires `d=2**n`;
`basis='wh'` uses the cyclic displacements above for arbitrary `d`.
Define the characteristic function, purity, and normalized weights

\[
\chi_U=\operatorname{Tr}(\rho U),\qquad
\gamma=\operatorname{Tr}(\rho^2),\qquad
\Xi_U=\frac{|\chi_U|^2}{d\gamma}.
\]

Operator orthogonality gives `sum_U |chi_U|²=d*gamma`, so `Xi` is a probability
distribution even for mixed states. It is not an ordinary WH-POVM outcome
distribution from an arbitrary fixed fiducial.

For pure states the implemented entropy is

\[
M_\alpha=H_\alpha(\Xi)-\log_b d,\qquad
H_\alpha(\Xi)=\frac{\log_b\sum_U\Xi_U^\alpha}{1-\alpha}.
\]

The `alpha=1` limit is Shannon entropy; logarithms default to base two.
The tensor-Pauli pure-state definition follows
[Leone, Oliviero and Hamma, *Stabilizer Rényi entropy*](https://arxiv.org/abs/2106.12587).
For example, pure qubit stabilizer states and Bell states have zero entropy,
and `( |0> + exp(i*pi/4)|1> )/sqrt(2)` has `M2=log2(4/3)`.

Passing `mixed=True` explicitly selects

\[
\widetilde M_\alpha=H_\alpha(\Xi)-\log_b(d\gamma),\qquad
\widetilde M_2=-\log_b\frac{\sum_U|\chi_U|^4}{d\gamma}.
\]

The mixed Pauli order-two expression appears in
[Oliviero, Leone and Hamma, *Magic-state resource theory for the ground state
of the transverse-field Ising model*](https://arxiv.org/abs/2205.02247).
The general-order formula is the explicitly defined diagnostic used here;
no additional resource-theoretic property is asserted for it.

This correction yields zero for the maximally mixed state and normalized
Pauli stabilizer-code projectors. **It is not a faithful mixed-state magic
monotone.** A simple counterexample is the free mixture
`rho=diag(3/4,1/4)`, for which

\[
\widetilde M_2=-\log_2\frac{1+(1/2)^4}{1+(1/2)^2}>0.
\]

The API deliberately names and documents this limitation. For a cyclic WH
basis, the same algebra defines a cyclic-qudit diagnostic; its value must not
be reported as tensor-Pauli magic for the encoded hardware qubits.

### Tomographic entropy route

`entropy_experiment` measures an IC POVM, reconstructs `rho`, then evaluates
the chosen characteristic function. It supports both operator conventions
and arbitrary positive finite Rényi orders. With finite shots, use
`mixed=True` and consider `analyze(..., physical=True)`: even a pure target
does not normally yield an exactly pure reconstructed matrix. `mixed=False`
rejects mixed estimates rather than silently changing the definition.

Entropy is nonlinear in estimated probabilities and generally biased. Physical
projection changes that bias. No bootstrap interval, debiasing procedure, or
maximum-likelihood fit is performed automatically.

### Direct cyclic-WH overlap route

`wh_entropy_experiment` prepares `rho` and its computational-basis conjugate
`rho*`, then applies a generalized Bell measurement. Its probabilities are

\[
s_{a,b}=\frac1d\operatorname{Tr}(\rho D_{a,b}\rho D_{a,b}^\dagger),
\qquad s_{0,0}=\frac\gamma d.
\]

Preparing the conjugate requires a known preparation description. There is
no claim of a universal physical conjugation channel for unknown inputs.
Mixed copies are independently purified, leaving both environments unmeasured.
The two logical registers occupy `2*ceil(log2(d))` measured qubits for `d>=2`;
output labels are still packed contiguously as `a*d+b`.

Here is a normalization check and derivation of the estimator. Expand `rho`
in WH operators. Conjugation by a displacement multiplies each component by
a symplectic Fourier character, giving

\[
s_g=\frac1{d^2}\sum_h |\chi_h|^2 e^{i\theta(g,h)}.
\]

The sign of the character depends on the displacement convention; Parseval's
identity does not. There are `d²` characters, so

\[
\sum_g s_g^2=\frac1{d^2}\sum_h|\chi_h|^4,\qquad
\widetilde M_2=-\log_b\frac{\sum_g s_g^2}{s_0}.
\]

`wh_entropy_from_overlaps` evaluates this expression directly. It requires no
IC fiducial or matrix reconstruction. The purity estimate `d*n_0/N` is
unbiased for independent ideal shots. The squared-frequency sum, ratio, and
logarithm are not unbiased; the implementation reports their plug-in values.
For comparison, `sum n_k*(n_k-1)/(N*(N-1))` would be unbiased for `sum s_k²`
when `N>1`, but substituting it would not remove ratio/log bias. That different
estimator is not silently used here.

Zero observed identity probability makes the ratio undefined and is rejected.
Sampling can produce negative entropy estimates or violate the exact bounds
`1/d² <= s_0 <= 1/d` and `s_k <= s_0`. Values are not clipped. Optional
`check_physical=True` checks those necessary bounds for exact/model data.

## Sky/ground Born-rule constraints

Let `E_i` be a minimal IC reference POVM and `sigma_k` its chosen reference
preparations. Let `B_j` be any ground POVM. Define

\[
p_i=\operatorname{Tr}(E_i\rho),\quad q_j=\operatorname{Tr}(B_j\rho),\quad
P_{ik}=\operatorname{Tr}(E_i\sigma_k),\quad
C_{jk}=\operatorname{Tr}(B_j\sigma_k).
\]

The columns of `P` and `C` are normalized probability vectors. If the reference
states span operator space, write `rho=sum_k a_k sigma_k`. Then `p=P a`,
`q=C a`, and consequently

\[
q=C P^{-1}p.
\]

For SIC projectors, `P=(I+J/d)/(d+1)` and
`P^-1=(d+1)I-J/d`, where every entry of `J` is one. Hence

\[
q_j=\sum_i\left((d+1)p_i-\frac1d\right)C_{ji}.
\]

These signed affine weights are intentional. Negative predictions from
inconsistent data are retained as diagnostics. They must not be replaced by
ordinary probabilities or normalized away.

The builder collects `2*S+2*d²` circuits for `S` input states: sky and direct
ground measurements for each state, plus reference and ground measurements
for each reference preparation. Its analyzed arrays are

| Array | Shape |
| --- | --- |
| `sky` | `(S,d²)` |
| `ground`, `prediction`, `residual` | `(S,ground_outcomes)` |
| `conditional` | `(ground_outcomes,d²)` |
| `reference` | `(d²,d²)` |

`residual` means `ground-prediction`. Reported norms and negative predicted
mass expose inconsistencies across preparations and experiments. The inverse
is estimated from calibration in this workflow; ill-conditioned calibration
can amplify noise, and a rank-deficient matrix is rejected. The inverse itself
is nonlinear in calibration frequencies, so a zero-mean shot error does not
imply an unbiased final residual.

`born_rule_diagnostics` can propagate independent multinomial sky, ground, and
conditional-table errors **when the reference inverse is treated as fixed**.
For a distribution `p` and `N` independent shots,
`Cov(p_hat)=(diag(p)-p p.T)/N`. Propagation uses `C P^-1`; conditional-column
covariances are weighted by `a_i²`. A fitted reference inverse, shared
calibration, cross-experiment drift, and correlated shots require additional
uncertainty analysis. The high-level sky/ground report does not claim those
uncertainties or produce a significance p-value.

A low residual can occur when devices share a consistent imperfect model.
A high residual can arise from drift, sampling, bad calibration, preparation
errors, or measurement errors. Repeating across diverse states and ground
measurements makes the battery more informative, but the constraint alone
does not distinguish those causes. The interpretation as a relation among
probability assignments follows the cited
[Fuchs–Schack formulation](https://arxiv.org/abs/0906.2187).

## Triple products and reconstruction of the state space

Throughout the library,

\[
T_{ijk}=\operatorname{Tr}(\Pi_i\Pi_j\Pi_k).
\]

Cyclic permutations leave `T` invariant; reversing the order conjugates it.
Repeated-index triples are real. `triple_products` computes the full complex
tensor from normalized kets or density matrices. SIC reconstruction requires
**unscaled SIC projectors**, not the effects `Pi/d`.

### Measuring real and imaginary triples

A controlled left cycle sends `|a,b,c>` to `|b,c,a>`. It is implemented by
controlled swaps of registers `(1,3)` followed by `(1,2)`. For three pure
inputs its expectation is

\[
\langle\psi_1|\psi_2\rangle
\langle\psi_2|\psi_3\rangle
\langle\psi_3|\psi_1\rangle
=\operatorname{Tr}(\rho_1\rho_2\rho_3).
\]

Linearity extends this to mixed inputs; purification environments are left
untouched by the swaps. The Hadamard-test control measures X for the real
part and Y for the imaginary part. The latter uses `S^-1` followed by `H`
before computational measurement. Both estimates use `P(0)-P(1)` with that
sign convention. For example, `|0>`, `|+>`, and `|+i>` have triple product
`(1+i)/4`.

For a component with expectation `t`, the signed sample mean is unbiased and
has variance `(1-t²)/N`. Plug-in standard errors vanish at sample endpoints,
so they are not reliable confidence intervals for rare opposite outcomes.
Real and imaginary tests use independent circuits. By default the builder
measures two circuits for each unordered triple, giving
`2*binomial(m+2,3)` circuits for `m` states; selected triplets avoid that cost.
Tensor assembly enforces cyclic/conjugation symmetries and sets imaginary
parts of repeated-index triples to their known value zero. Measured raw
estimates remain available separately.

### Moments and pure-state constraints

Given SIC probabilities, put `a_i=(d+1)p_i-1/d`. Then

\[
\operatorname{Tr}(\rho^2)=\sum_{ij}a_i a_j G_{ij}
=d(d+1)\sum_i p_i^2-1,
\]

\[
\operatorname{Tr}(\rho^3)=\sum_{ijk}a_i a_j a_k\operatorname{Re}T_{ijk}.
\]

For a Hermitian trace-one operator, the two moment equations
`Tr(rho²)=Tr(rho³)=1` imply purity: the first bounds every eigenvalue by one
in magnitude, and `sum lambda²*(1-lambda)=0` then forces each eigenvalue to
be zero or one. The SIC probability version is discussed in
[Appleby, Ericsson and Fuchs, *Properties of QBist State Spaces*](https://arxiv.org/abs/0910.2750).
Mixed-state moment bounds alone do not characterize positivity in general
dimensions; a complete check follows.

### A complete positivity pencil from triples alone

Define `d²` matrices of size `d²` by

\[
A^{(i)}_{jk}=(d+1)T_{jik}-G_{jk},\qquad
L(p)=\sum_i p_i A^{(i)}.
\]

For normalized `p`, `sum_i Pi_i=dI` gives
`L(p)_{jk}=Tr(Pi_j rho Pi_k)`. If `X=sum_k x_k Pi_k`, then

\[
x^\dagger L(p)x=\operatorname{Tr}(X^\dagger\rho X).
\]

The projectors span operator space. Thus `rho>=0` implies `L(p)>=0`, and if
`rho` has a negative eigenvalue, choosing `X` to be its eigenprojector gives
a negative quadratic form. This proves the equivalence. Real coefficients
already span all Hermitian `X`, so `Re L(p)` gives the same positivity test.
Consequently **real triples suffice to reconstruct the probability state
space**, while imaginary triples retain additional operator-algebra
orientation information. This derivation describes the pencil implemented in
`triple_product_state_space`.

The reconstructed set is

\[
\{p:\ p_i\ge0,\ \sum_i p_i=1,\ L(p)\succeq0\}.
\]

`geometry` ray-shoots 2D/3D affine sections `p=center+B*y`. The center must be
strictly inside both the simplex and positive-definite matrix pencil.
A direction stops at whichever comes first: a simplex facet or a zero
matrix eigenvalue. Finite angular or mesh resolution approximates the
boundary between sampled points.

For experimental triples, the code Hermitianizes the pencil; it does not fit
a statistically consistent SIC algebra. Noise can produce an empty or
distorted feasible set. Moment and positivity outputs are diagnostics of the
measured tensor, not certificates that noisy data describe an exact SIC.
Reconstructing this probability region also does not recover a unique SIC
fiducial or Hilbert-space orientation.

## Resources, execution, and reproducibility

The implementation is an exact dense reference, useful for debugging and
small experiments. A dilation with `M=2**N` uses `O(M²)` matrix storage and
dense completion can take `O(M³)` time. Generic synthesis and dense state
preparation grow exponentially with qubit count. `compile_circuit` uses
cosine–sine/quantum Shannon decomposition, Schur-based block demultiplexing,
and Cirq target-gateset synthesis; the underlying construction is described
by [Shende, Bullock and Markov, *Synthesis of Quantum Logic Circuits*](https://arxiv.org/abs/quant-ph/0406176).
It does not place qubits, route connectivity, choose hardware, or validate
hardware calibration. Target synthesis can drop a circuit's physically
irrelevant overall phase.

For `d>=2`, a triple test uses `1+3*ceil(log2(d))` qubits for kets, plus one
same-sized environment per mixed input. A direct Bell-overlap test uses two
system registers, plus two environments for matrix inputs. Purification
makes exact state-vector simulation possible, but adds qubits. Channel/noise
simulation uses a density matrix: for `Q` total qubits it stores `4**Q`
complex entries rather than `2**Q`. Full SIC triple tensors and positivity
pencils store `O(d**6)` entries. Passing selected triplets and small section
meshes is often essential.

`simulate()` returns exact model distributions, not shot samples; `run()`
returns counts from an explicitly supplied sampler or a seeded local sampler.
The unified model permits a single ordinary terminal readout per circuit.
Invert masks, confusion maps, and classical feedforward are rejected because
they require additional decoding or dynamic measurement semantics. Explicit
channel/noise operations remain available, and noise applied through
`simulate(noise=...)` includes recorded idle qubits. Noise is inserted into
the original moment sequence and the exact distribution is taken at the
recorded readout: noise before it contributes, noise after it does not.
This also preserves actual empty idle moments. Backend samplers can use
different implicit noise timing or terminal-sampling optimizations. Use `noisy = experiment.with_noise(noise)` to save the inserted operations
and obtain the same model through `noisy.simulate()` and `noisy.run()` with
an otherwise noiseless simulator. The resulting circuits preserve terminal
readout and receive their own content fingerprints.

Saved experiments contain Cirq JSON, explicit qubit order, measurement layout,
analysis arrays, metadata, and a deterministic content fingerprint. Generated
results carry that fingerprint. Analysis rejects results belonging to another
experiment even if circuit names and dimensions coincide; editing arrays,
metadata, or compiled circuits changes the fingerprint. Save/load preserves
identity and verifies a saved checksum. This detects accidental mismatches,
not malicious artifact modification. Imported results without provenance
require explicit `allow_unbound=True` after independently checking their
origin and decoding. A mismatched supplied fingerprint is always rejected.

A compiled experiment is a new recorded circuit specification. Analyze its
results with that compiled experiment. Hardware samplers, noise models,
shot counts, and seeds belong in execution provenance; an experiment
fingerprint does not certify that a backend actually executed the saved
circuit. No code or hardware jobs are submitted merely by loading artifacts.
