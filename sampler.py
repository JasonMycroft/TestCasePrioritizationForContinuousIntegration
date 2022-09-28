import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_rows", None, "display.max_columns", None)


def randOversampling(data_df, samp_strat=0.30):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        oversample = RandomOverSampler(sampling_strategy=samp_strat)
        X_resampled, y_resampled = oversample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return pd.DataFrame(sampled_df)


def smoteOversampling(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        oversample = SMOTE(random_state=42, k_neighbors=1, sampling_strategy=0.50)
        X_resampled, y_resampled = oversample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return sampled_df


def nearMiss(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        undersample = NearMiss(version=2, n_neighbors=2, sampling_strategy=0.20)
        X_resampled, y_resampled = undersample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return sampled_df

def adasynSampling(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        oversample = ADASYN(random_state=42, n_neighbors=1, sampling_strategy=0.30)
        X_resampled, y_resampled = oversample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)

    return sampled_df