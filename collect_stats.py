import pandas as pd
import os


results_dir = "C:\\Users\\mycro\\Downloads\\datasets\\results" #results directory
results_folders = os.listdir(results_dir)
stats_df = pd.DataFrame()
# gather apfdc info from each dataset
all_results_df = pd.DataFrame()
for results_folder in results_folders:
	folder = os.path.join(results_dir, results_folder)
	if os.path.isdir(folder):
		# read apfdc values
		results_path = os.path.join(folder, "results.csv")
		results_df = pd.read_csv(results_path, usecols=['apfdc'])
		apfdc_stats_df = results_df['apfdc'].describe()
		# write apfdc info to dataframe
		row = {'dataset': results_folder, 'mean_apfdc': apfdc_stats_df['mean'], 'std_apfdc': apfdc_stats_df['std'], '50%_apfdc': apfdc_stats_df['50%']}
		stats_df = stats_df.append(row, ignore_index=True)
		# accumulate apfdc values for all datasets
		all_results_df = all_results_df.append(results_df, ignore_index=True)
# write stats for all datasets into same folder
all_apfdc_stats_df = all_results_df['apfdc'].describe()
row = {'dataset': 'ALL', 'mean_apfdc': all_apfdc_stats_df['mean'], 'std_apfdc': all_apfdc_stats_df['std'], '50%_apfdc': all_apfdc_stats_df['50%']}
stats_df = stats_df.append(row, ignore_index=True)
stats_df = stats_df[['dataset', 'mean_apfdc', 'std_apfdc', '50%_apfdc']]
stats_df.to_csv(os.path.join(results_dir, "apfdc_stats.csv"), index=False)