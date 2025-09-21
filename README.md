## circ_sic

Implementing SIC's in cirq.

After installing `pip install -e`, to perform a battery of sky/ground experiments using the $d=4$ SIC fiducial, run

```python sky_ground_run.py -dataset_id example -run_type clean/noisy/real -wh_implementation ak/simple -n_shots 10000```

`clean` means a noiseless simulation, `noisy` uses a noise simulator, `real` should run it on `willow` itself.  

You can choose to use the Arthurs-Kelly (`ak`) implementation or the simple WH-POVM implementation (`simple`). 

The four experiments are then run

1. Preparing each of the SIC states in turn, and then performing a SIC measurement. We can gather this data into a matrix of probabilities $P_{ij} = P(R_i | R_j)$.
2. Preparing each of the computational basis states $\{ \Pi_i = |i\rangle\langle i| \}$, and then performing a SIC measurement. We can gather the probabilities into a matrix $p_{ij} = P(R_j | \Pi_i)$.
3. Preparing each of the SIC states, and performing a computational basis measurement, yielding a matrix of probabilities $C_{ij} = P(\Pi_i | R_j)$.
4. Preparing each of the computational basis states, and performing a computational basis measurement, yielding probabilities $q_{ij} = P(\Pi_i |\Pi_j)$.

The data is stored in JSON files of the form

`data/sky_ground/dataset_id/description__wh_implementation__willow_pink__run_type__n_shots__qubits.json`

which can be loaded via

```
experiment_name = "sky_ground"
base_dir = f'../experiments/data/{experiment_name}'

records = recirq.load_records(dataset_id="9_18_25", base_dir=base_dir)
results = query_records(records, {"dataset_id": "9_18_25",
                                  "processor_id": "willow_pink",
                                  "run_type": "noisy",
                                  "wh_implementation": "ak",
                                  "n_shots": 100000,
                                  "d": 4,\
                                  "description": "characterize_wh_reference_device"})
```

The possible descriptions are `characterize_wh_reference_device`, `wh_povm_on_basis_states`, `basis_measurement_on_wh_states`, `basis_measurement_on_basis_states`.

`query_records` returns a list of records which match the description. Then one may access `result["P"]`, `result["p"]`, `results["C"]`, and `results["q"]` in each of the four cases to check whether $C \Phi p = q$, where $\Phi=P^{-1}$.

