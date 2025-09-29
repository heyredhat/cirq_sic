## circ_sic

Implementing SIC's in cirq.

After installing `pip install -e`, to perform a battery of sky/ground experiments, run

```python sky_ground_run.py -dataset_id example -run_type clean/noisy/real -wh_implementation ak/simple -n_shots 10000 -flag numerical_sic```

`clean` means a noiseless simulation, `noisy` uses a noise simulator, `real` should run it on `willow` itself.  

You can choose to use the Arthurs-Kelly (`ak`) implementation or the simple WH-POVM implementation (`simple`). Change `flag` to `d4_monomial` if you want in dimension 4.

The four experiments are then run

1. Preparing each of the SIC states in turn, and then performing a SIC measurement. We can gather this data into a matrix of probabilities $P_{ij} = P(R_i | R_j)$.
2. Preparing each of the computational basis states $\{ \Pi_i = |i\rangle\langle i| \}$, and then performing a SIC measurement. We can gather the probabilities into a matrix $p_{ij} = P(R_j | \Pi_i)$.
3. Preparing each of the SIC states, and performing a computational basis measurement, yielding a matrix of probabilities $C_{ij} = P(\Pi_i | R_j)$.
4. Preparing each of the computational basis states, and performing a computational basis measurement, yielding probabilities $q_{ij} = P(\Pi_i |\Pi_j)$.





