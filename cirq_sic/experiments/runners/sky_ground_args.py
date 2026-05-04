with open("args.txt", "w") as f:
    n_shots = 100000
    fiducial_description = "numerical_sic"
    for n in [1,2,3]:            
        d = 2**n
        for run_type in ["clean", "noisy"]:
            for optimizer in ["cirq", "bqskit_server_4"]:
                for wh_implementation in ["ak", "simple", "msimple"]:
                    str = f"-dataset_id 11_2_25 -processor_id willow_pink -run_type {run_type} -n_shots {n_shots} -optimizer {optimizer} -d {d} -fiducial_description {fiducial_description} -wh_implementation {wh_implementation}"
                    f.write(str+"\n")
