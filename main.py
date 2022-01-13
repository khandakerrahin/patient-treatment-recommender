# Description: Build a Therapy Recommendation System
# Author: Shaker Mahmud Khandaker
# Course: Data Mining
# Matricola: 229221


# imports
import json
import random
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

print(
    '# Description: Build a Therapy Recommendation System\n# Author: Shaker Mahmud Khandaker\n# '
    'Course: Data Mining\n# Matricola: 229221\n')

dataset = 'dataset_sample.json'


def get_current_datetime_string():
    # datetime object containing current date and time
    now = datetime.now()

    # print("now =", now)

    # ddmmYY_HMS
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    # print("date and time =", dt_string)

    return dt_string


# print summary of dataset
def print_dataset_summary(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # print dataset summary
    print('Dataset Summary:\n')
    print('# Conditions: ', len(data['Conditions']))
    print('# Therapies: ', len(data['Therapies']))
    print('# Patients: ', len(data['Patients']))
    # print('Trials: ', len(data['Patients'][2]['trials']))

    trial_counts = 0
    for patient in data['Patients']:
        # print('Trials: ', len(patient['trials']))
        trial_counts += len(patient['trials'])

    print('# Trials: ', trial_counts)


def get_keys_string(data):
    keys_string = ''
    for x in data[0].keys():
        keys_string = keys_string + x + ','

    keys_string = keys_string[:-1]
    # print(keys_string)

    return keys_string


def create_conditions_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    df = pd.json_normalize(data['Conditions'])

    df.to_csv(r'DF_Conditions_temp.csv', index=None)


def create_therapies_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    df = pd.json_normalize(data['Therapies'])

    df.to_csv(r'DF_Therapies_temp.csv', index=None)


def create_patients_cases_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # trials
    df_trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='_')

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    # print(df_trials)
    # print(df_conditions)

    merged_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])
    # print(merged_df)
    merged_df.to_csv(r'DF_Patients_cases.csv', index=None)


def create_patients_similarity_by_condition(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    df_conditions = df_conditions.groupby(['_kind', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_patients_by_condition.csv')


def create_patients_similarity_by_therapies(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # therapy
    df_therapy = pd.json_normalize(data['Patients'], "trials", "id", errors='ignore', record_prefix='_')
    # print(df_conditions)

    df_therapy = df_therapy.groupby(['_therapy', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_therapy.corr(method='pearson')
    display(similarity_df.head(10))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_patients_by_therapy.csv')


def create_conditions_similarity_by_patients(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    df_conditions = df_conditions.groupby(['id', '_kind']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_conditions_by_patients.csv')


def create_therapies_similarity_by_patients(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # therapy
    df_therapy = pd.json_normalize(data['Patients'], "trials", "id", errors='ignore', record_prefix='_')

    df_therapy = df_therapy.groupby(['id', '_therapy']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_therapy.corr(method='pearson')
    display(similarity_df.head(10))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_therapies_by_patients.csv')


def create_patients_therapies_similarity_by_success(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # therapy
    df_therapy = pd.json_normalize(data['Patients'], "trials", "id", errors='ignore', record_prefix='_')

    patient_therapy_matrix = df_therapy.pivot_table(index='id', columns='_therapy', values='_successful')

    # patient_therapy_matrix = patient_therapy_matrix.fillna(0)
    # patient_therapy_matrix = patient_therapy_matrix.fillna(condition_therapy_matrix.mean(axis=0))
    display(patient_therapy_matrix.head(20))

    patient_therapy_matrix.to_csv(r'patient_therapy_matrix.csv')

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(patient_therapy_matrix, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'patient_therapy_similarity.csv')


def create_conditions_therapies_similarity_by_success(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # trials
    df_trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='_')

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    # print(df_trials)
    # print(df_conditions)

    merged_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])
    # print(merged_df)

    condition_therapy_matrix = merged_df.pivot_table(index='_kind', columns='_therapy', values='_successful')

    # condition_therapy_matrix = condition_therapy_matrix.fillna(0)
    # condition_therapy_matrix = condition_therapy_matrix.fillna(condition_therapy_matrix.mean(axis=0))
    display(condition_therapy_matrix.head(20))

    condition_therapy_matrix.to_csv(r'condition_therapy_matrix.csv')

    # similarity_df = condition_therapy_matrix.corr(method='pearson')
    # display(similarity_df.head(10))

    # print(check_if_below_average(condition_therapy_matrix, 0.24))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(condition_therapy_matrix, 10)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'conditions_therapies_similarity.csv')


def create_therapies_similarity_by_therapies(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    df_conditions = pd.json_normalize(data['Therapies'])

    print(df_conditions)

    df_conditions = df_conditions.groupby(['type', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))
    similarity_df.to_csv(r'similar_therapies_scores.csv')

    # print(check_if_below_average(similarity_df, 0.24))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_therapies_by_therapies.csv')


def create_conditions_similarity_by_conditions(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    df_conditions = pd.json_normalize(data['Conditions'])

    print(df_conditions)

    df_conditions = df_conditions.groupby(['type', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))
    similarity_df.to_csv(r'similar_conditions_scores.csv')

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'similar_conditions_by_conditions.csv')


def get_min_df(dataframe, index):
    return dataframe.iloc[index].min()


def get_max_df(dataframe, index):
    return dataframe.iloc[index].max()


def check_if_below_average(dataframe, value):
    min_value = get_min_df(dataframe, 0)
    max_value = get_max_df(dataframe, 0)

    print('Min: ', min_value)
    print('Max: ', max_value)

    average_value = (min_value+max_value)/2

    # print(average_value)

    return value < average_value


def find_n_neighbours(df, n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                      .iloc[:n].index,
                                      index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
    return df


# get unique file name
def get_unique_file_name(prefix, suffix):
    # return 'dataframe_temp_' + get_current_datetime_string() + '.csv'
    return prefix + get_current_datetime_string() + suffix


def get_therapy_recommendation():
    return 0


def main():
    # print file summary
    # print_dataset_summary(dataset)

    # create_conditions_dataframe(dataset)
    # create_therapies_dataframe(dataset)
    # create_patients_cases_dataframe(dataset)

    # create_patients_similarity_by_condition(dataset)
    # create_patients_similarity_by_therapies(dataset)
    # create_conditions_similarity_by_patients(dataset)
    # create_therapies_similarity_by_patients(dataset)
    # create_conditions_similarity_by_conditions(dataset)
    # create_therapies_similarity_by_therapies(dataset)
    # create_conditions_therapies_similarity_by_success(dataset)    # top successful therapies for a condition
    create_patients_therapies_similarity_by_success(dataset)    # top successful therapies for a patient


if __name__ == '__main__':
    main()

# PLOTTING CODES
# trial_df = pd.read_csv("DF_Patients_temp.csv", encoding="unicode_escape")

# display(trial_df.head())

# display(trial_df.describe())
# display(pd.DataFrame(trial_df.groupby('_kind')['_successful'].mean()))

# success_rate = pd.DataFrame(trial_df.groupby('_kind')['_successful'].mean())
# # display(success_rate.head())
# success_rate['count'] = trial_df.groupby('_kind')['_successful'].count()
# # display(success_rate.head())
#
# display(sns.jointplot(x='_successful', y='count', data=success_rate))
#
# plt.show()
# therapy_user_matrix = trial_df.pivot_table(index='id', columns='_therapy', values='_successful')
# display(therapy_user_matrix.head())

# therapy_user_matrix.to_csv(r'RAHIN-01.csv', index=None)

# therapy_user_matrix = trial_df.pivot_table(index='_kind', columns=['id'], values='_successful')
# therapy_user_matrix = therapy_user_matrix.fillna(0)
# display(therapy_user_matrix.head())
#
# therapy_user_matrix.to_csv(r'therapy_user_matrix.csv')
#
# item_similarity_df = therapy_user_matrix.corr(method='pearson')
# display(item_similarity_df.head(50))
#
# item_similarity_df.to_csv(r'item_similarity_df.csv')
# print(item_similarity_df['c_00056'])
