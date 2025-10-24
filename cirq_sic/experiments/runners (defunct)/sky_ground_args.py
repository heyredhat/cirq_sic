with open("args.txt", "w") as f:
    n_shots = 100000
    for n in [1,2,3]:            
        d = 2**n
        for run_type in ["clean", "noisy"]:
            for wh_implementation in ["ak", "simple"]:
                str = f"-dataset_id d{d} -run_type {run_type} -wh_implementation {wh_implementation} -n_shots {n_shots} -d {d}"
                if d != 4:
                    f.write(str+" -flag numerical_sic\n")
                else:
                    for flags in [" -flag d4_monomial"," -flag numerical_sic"]:
                        f.write(str+flags+"\n")
                    
                        