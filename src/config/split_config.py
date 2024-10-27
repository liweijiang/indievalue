

def get_split_config(expt_name):
    if expt_name == "WorldValuesBench_probe":
        return {
            "expt_name": "WorldValuesBench_probe",
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        }