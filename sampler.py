import random
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_rows", None, "display.max_columns", None)


def randOversampling(data_df):
    #print("Random Oversampling")
    data = data_df
    # print(data.head())
    #print(data['Verdict'].value_counts())
    data.loc[(data.Verdict == 2), 'Verdict'] = 1
    majority = data['Verdict'].value_counts(ascending=False)[0]
    # print(majority)
    oversample = RandomOverSampler(
        sampling_strategy=0.50)
    X_resampled, y_resampled = oversample.fit_resample(data.drop(['Verdict'], axis=1), data['Verdict'])
    data_oversampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    data_oversampled = data_oversampled.reindex(columns=data.columns.values.tolist())
    #print(data_oversampled['Verdict'].value_counts())
    # print(data_oversampled.head())
    sorted = data_oversampled.sort_values('Build')
    #sorted.to_csv('random13.csv', index=False)
    return sorted


def smoteOversampling(data_df):
    #print("SMOTE Oversampling")
    data = data_df
    buildIDS = data['Build'].unique().tolist()
    #print(buildIDS)
    # print(data.head())
    data.loc[(data.Verdict == 2), 'Verdict'] = 1
    #print(data['Verdict'].value_counts())
    #majority = data['Verdict'].value_counts(ascending=False)[0]
    # print(majority)
    oversample = SMOTE(random_state=42, sampling_strategy=0.50)
    X_resampled, y_resampled = oversample.fit_resample(data.drop(['Verdict'], axis=1), data['Verdict'])
    data_oversampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    data_oversampled = data_oversampled.reindex(columns=data.columns.values.tolist())
    #print(data_oversampled['Verdict'].value_counts())
    #print(random.choice(buildIDS))
    #data_oversampled.loc[~data_oversampled.Build.isin(buildIDS), 'Build'] = random.choice(buildIDS)
    for index, row in data_oversampled.iterrows():
        if row.Build not in buildIDS:
            data_oversampled.loc[index, 'Build'] = random.choice(buildIDS)
    sorted = data_oversampled.sort_values('Build')
    #sorted.to_csv('smote13.csv', index=False)
    return sorted


def nearMiss(data_df):
    #print("Near Miss Undersampling")
    data = data_df
    # print(data.head())
    data.loc[(data.Verdict == 2), 'Verdict'] = 1
    #print(data['Verdict'].value_counts())
    #majority = data['Verdict'].value_counts()[0]
    # print(majority)
    undersample = NearMiss(version=2, n_neighbors=3, sampling_strategy=0.70)
    X_resampled, y_resampled = undersample.fit_resample(data.drop(['Verdict'], axis=1), data['Verdict'])
    data_undersampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    data_undersampled = data_undersampled.reindex(columns=data.columns.values.tolist())
    #print(data_undersampled['Verdict'].value_counts())
    sorted = data_undersampled.sort_values('Build')
    #sorted.to_csv('nearmiss1.csv', index=False)
    return sorted

def adasynSampling(data_df):
    #print("ADASYN Oversampling")
    data = data_df
    buildIDS = data['Build'].unique().tolist()
    #print(buildIDS)
    # print(data.head())
    #print(data['Verdict'].value_counts())
    #majority = data['Verdict'].value_counts(ascending=False)[0]
    # print(majority)
    oversample = ADASYN(random_state=42, n_neighbors=5,
                       sampling_strategy=0.50)
    X_resampled, y_resampled = oversample.fit_resample(data.drop(['Verdict'], axis=1), data['Verdict'])
    data_oversampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    data_oversampled = data_oversampled.reindex(columns=data.columns.values.tolist())
    #print(data_oversampled['Verdict'].value_counts())
    #print(random.choice(buildIDS))
    # data_oversampled.loc[~data_oversampled.Build.isin(buildIDS), 'Build'] = random.choice(buildIDS)
    for index, row in data_oversampled.iterrows():
        if row.Build not in buildIDS:
            data_oversampled.loc[index, 'Build'] = random.choice(buildIDS)
    sorted = data_oversampled.sort_values('Build')
    #sorted.to_csv('adasyn13.csv', index=False)
    return sorted

def smoteNMSampling(data_df):
    smoted = smoteOversampling(data_df)
    nm = nearMiss(smoted)
    return nm




if __name__ == '__main__':
    df = pd.read_csv("dataset13.csv")
    smoteOversampling(df)
