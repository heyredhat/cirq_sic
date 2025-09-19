with open("sky_ground_args.txt", "w") as f:
    for run_type in ["clean", "noisy"]:
        for n_shots in [1000, 5000, 10000, 20000, 50000, 100000]:
            for wh_implementation in ["ak", "simple"]:
                f.write(f"-dataset_id 9_18_25 -run_type {run_type} -wh_implementation {wh_implementation} -n_shots {n_shots}\n")