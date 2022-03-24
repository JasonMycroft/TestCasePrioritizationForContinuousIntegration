import pandas as pd
import numpy as np
import os
import scipy.stats as st

results_dir = "G:\\Downloads\\!datasets\\!results\\RQ2.1"
frames = []
for filename in os.listdir(results_dir):
    results_path = os.path.join(results_dir, filename)
    results_df = pd.read_csv(results_path, usecols=['mean_apfdc'], index_col=False)
    results_df.drop(results_df.tail(1).index,inplace=True)
    frames.append(results_df['mean_apfdc'].values)
for frame in frames:
    stat, pvalue = st.normaltest(frame)
    if pvalue > 0.05:
        print('not normally distributed')
    else:
        print('normally distributed')