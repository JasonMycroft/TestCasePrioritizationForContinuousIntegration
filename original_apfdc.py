import pandas as pd
import os
from pathlib import Path

def compute_apfdc(build):
    if len(build) <= 1:
        return 1.0
    m = len(build[build["Verdict"] > 0])
    costs = build["Duration"].values.tolist()
    failed_costs = 0.0
    for tfi in build[build["Verdict"] > 0].index:
        tfi = tfi - build.index[0] # offset to account for groupby
        failed_costs += sum(costs[tfi:]) - (costs[tfi] / 2)
    apfdc = failed_costs / (sum(costs) * m)
    return float("{:.3f}".format(apfdc))

datasets_dir = "C:\\Users\\mycro\\Downloads\\datasets" # datasets directory
results_dir = os.path.join(datasets_dir, "results") # results directory
Path(results_dir).mkdir(parents=True, exist_ok=True)

builds_for_training = 10
i = 0
while i < 25:
    i = i+1
    dataset_path = os.path.join(datasets_dir, f"dataset{i}.csv")
    if not os.path.isfile(dataset_path):
        print(f"No dataset{i}.csv found in the output directory.")
        continue
    #print(f"##### Computing APFDc for dataset{i}.csv #####")
    results = {"build": [], "apfdc": []}
    dataset_df = pd.read_csv(dataset_path)[['Build', 'Verdict', 'Duration']]
    build_ids = dataset_df['Build'].unique().tolist()
    builds = dict(tuple(dataset_df.groupby('Build')))
    # skip first 10 builds
    for id in build_ids[builds_for_training:]:
        apfdc = compute_apfdc(builds[id])
        results["build"].append(id)
        results["apfdc"].append(apfdc)
    results_df = pd.DataFrame(results)
    out_path = os.path.join(results_dir, f"results{i}")
    Path(out_path).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(os.path.join(out_path, f"results.csv"), index=False)
import collect_stats