import argparse

import cirq
import numpy as np

from cirq_sic import *

####################################################################################

experiment_name = "sky_ground"
base_dir = f'data/{experiment_name}'

####################################################################################

def main():
    parser = argparse.ArgumentParser(prog="sky_ground")
    parser.add_argument("-dataset_id", type=str, required=True)
    parser.add_argument("-processor_id", type=str, required=True)
    parser.add_argument("-run_type", type=str, required=True)
    parser.add_argument("-n_shots", type=int, required=True)
    parser.add_argument("-optimizer", type=str, required=True)
    parser.add_argument("-d", type=int, required=True)
    parser.add_argument("-fiducial_description", type=str, required=True)
    parser.add_argument("-wh_implementation", type=str, required=True)
    specs = vars(parser.parse_args())
    specs["qubits"] = get_wh_qubits(specs["d"], specs["wh_implementation"])
    if specs["fiducial_description"] == "numerical_sic":
        specs["fiducial"] = load_sic_fiducial(specs["d"])
    elif specs["fiducial_description"] == "rand_ket":
        specs["fiducial"] = rand_ket(specs["d"])
    elif specs["fiducial_description"] == "d4_monomial":
        specs["fiducial_circuit"] = d4_sic_fiducial

    run_sky_ground_tasks(specs, base_dir=base_dir)
    sky_ground_images(specs, base_dir=base_dir, img_dir=f"{base_dir}/{specs["dataset_id"]}/d{specs["d"]}/img")

if __name__ == "__main__":
    main()