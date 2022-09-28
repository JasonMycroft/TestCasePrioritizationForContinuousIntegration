import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_rows", None, "display.max_columns", None)


def randOversampling(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]
        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.50) * zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        oversample = RandomOverSampler(sampling_strategy=0.50)
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
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]
        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        if one < 2:
            randos = RandomOverSampler(sampling_strategy={1: 2})
            X_temp, y_temp = randos.fit_resample(cur_build.drop(['Verdict'], axis=1),cur_build['Verdict'])
            temp_build = pd.concat([pd.DataFrame(X_temp), pd.DataFrame(y_temp)], axis=1)
            temp_build = temp_build.reindex(columns=data.columns.values.tolist())
            cur_build = temp_build
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.50)*zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
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
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]
        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.20)*zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
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
        occ = cur_build['Build'].value_counts()[build]
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        print(build)
        if occ == 1 or occ == one:
            continue
        if one < 5:
            continue
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.50)*zero)) <= one:
            continue
        oversample = ADASYN(random_state=42, n_neighbors=1, sampling_strategy=0.50)
        X_resampled, y_resampled = oversample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return sampled_df

def randUndersampling(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]
        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.20) * zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        undersample = RandomUnderSampler(sampling_strategy=0.20)
        X_resampled, y_resampled = undersample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return pd.DataFrame(sampled_df)

def smoteBorderline(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]
        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        if one < 2:
            randos = RandomOverSampler(sampling_strategy={1: 2})
            X_temp, y_temp = randos.fit_resample(cur_build.drop(['Verdict'], axis=1),cur_build['Verdict'])
            temp_build = pd.concat([pd.DataFrame(X_temp), pd.DataFrame(y_temp)], axis=1)
            temp_build = temp_build.reindex(columns=data.columns.values.tolist())
            cur_build = temp_build
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.50)*zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        oversample = BorderlineSMOTE(random_state=42, k_neighbors=1, m_neighbors=2, sampling_strategy=0.50, kind='borderline-2')
        X_resampled, y_resampled = oversample.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return sampled_df

def randOverUnder(data_df):
    data = data_df
    sampled_df = pd.DataFrame(columns=data.columns.values.tolist())
    buildIDS = data['Build'].unique().tolist()

    data.loc[(data.Verdict == 2), 'Verdict'] = 1

    for build in buildIDS:
        cur_build = data.loc[data.Build == build]
        occ = cur_build['Build'].value_counts()[build]
        one = cur_build['Verdict'].value_counts()[1]

        if occ == 1 or occ == one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        zero = cur_build['Verdict'].value_counts()[0]
        one = cur_build['Verdict'].value_counts()[1]
        if int(round((0.5) * zero)) <= one:
            sampled_df = sampled_df.append(cur_build, ignore_index=True)
            continue
        oversample = RandomOverSampler(sampling_strategy=0.5)
        undersample = RandomUnderSampler(sampling_strategy=0.60)
        pipeline = Pipeline(steps=[('o', oversample), ('u', undersample)])

        X_resampled, y_resampled = pipeline.fit_resample(cur_build.drop(['Verdict'], axis=1), cur_build['Verdict'])
        sampled_build = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        sampled_build = sampled_build.reindex(columns=data.columns.values.tolist())
        sampled_build.loc[sampled_build.Build != build, 'Build'] = build
        sampled_df = sampled_df.append(sampled_build, ignore_index=True)
    return pd.DataFrame(sampled_df)