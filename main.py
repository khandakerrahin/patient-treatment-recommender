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
    '# Description: A Therapy Recommendation System\n# Author: Shaker Mahmud Khandaker\n# '
    'Course: Data Mining\n# Matricola: 229221\n')

# dataset counts
condition_count = 0
therapy_count = 0
patient_count = 0
trial_count = 0


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

    global therapy_count, condition_count, patient_count, trial_count

    condition_count = len(data['Conditions'])
    therapy_count = len(data['Therapies'])
    patient_count = len(data['Patients'])
    trial_count = 0

    for patient in data['Patients']:
        # print('Trials: ', len(patient['trials']))
        trial_count += len(patient['trials'])

    # print dataset summary
    print('Dataset Summary:\n')
    print('# Conditions: ', condition_count)
    print('# Therapies: ', therapy_count)
    print('# Patients: ', patient_count)
    print('# Trials: ', trial_count, '\n')


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


def create_patients_trials_dataframe(filename):
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
    print(df_conditions)

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
    print(df_conditions)

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


def create_patients_therapies_similarity_by_success(filename, patient_id):
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
    similar_neighbours = find_n_neighbours(patient_therapy_matrix, therapy_count)
    display(similar_neighbours.head(20))

    similar_neighbours.to_csv(r'patient_therapy_similarity.csv')

    rowData = similar_neighbours.loc[int(patient_id), :]
    print('Top Therapies for ' + patient_id + ': ')
    for items in rowData:
        print(items)


def global_therapies_recommendation_for_conditions_by_success(filename, condition):
    trials_df = pd.read_csv("DF_Patients_cases.csv", encoding="unicode_escape")

    condition_therapy_matrix = trials_df.pivot_table(index='_kind', columns='_therapy', values='_successful')

    # condition_therapy_matrix = condition_therapy_matrix.fillna(0)
    # condition_therapy_matrix = condition_therapy_matrix.fillna(condition_therapy_matrix.mean(axis=0))
    # display(condition_therapy_matrix.head(20))

    # condition_therapy_matrix.to_csv(r'condition_therapy_matrix.csv')

    # similarity_df = condition_therapy_matrix.corr(method='pearson')
    # display(similarity_df.head(10))

    # print(check_if_below_average(condition_therapy_matrix, 0.24))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(condition_therapy_matrix, 5)
    # display(similar_neighbours.head(20))

    # similar_neighbours.to_csv(r'conditions_therapies_similarity.csv')
    # similar_neighbours[condition]

    trials_df = trials_df[trials_df['_kind'] == condition]

    trials_df = trials_df.groupby('_therapy')['_successful'].mean()

    # print('FILTERED AND GROUPED FOR '+condition)
    # print(trials_df)

    rowData = similar_neighbours.loc[condition, :]
    print('Top Therapies for ' + condition + ': ')
    for items in rowData:
        try:
            print(items, trials_df[items])
        except:
            break


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
                                      index=['top_{}'.format(i) for i in range(1, n + 1)]), axis=1)
    return df


# get unique file name
def get_unique_file_name(prefix, suffix):
    # return 'dataframe_temp_' + get_current_datetime_string() + '.csv'
    return prefix + get_current_datetime_string() + suffix


def filter_df_features(filename):
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

    # _condition, _end, _id_x, _start, _cured, _diagnosed, _id_y, _isCured, _isTreated, name

    columns = ['id', '_kind', '_therapy', '_successful']

    # print(merged_df[columns])

    # merged_df.to_csv(r'DF_Patients_cases.csv', index=None)
    # merged_df[columns].to_csv(r'DF_filtered_Patients_cases.csv', index=None)

    return merged_df[columns]


def get_useful_features(data):
    useful_features = []

    for c in range(0, data.shape[0]):
        useful_features.append(data['id'][c] + ' ' + data['_kind'][c] + ' ' + data['_therapy'][c] + ' ' + data['_successful'][c])

    return useful_features


def get_condition_id(filename, patient_condition_id):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')
    # print(df_conditions)
    row_cond = df_conditions.loc[df_conditions['_id'] == patient_condition_id]
    cond_id = row_cond['_kind'].values[0]
    # print('BLA ConditionID: ', cond_id)
    return cond_id


def get_patient_therapies_ranking(patient_condition_id):
    trials_df = pd.read_csv("DF_Patients_cases.csv", encoding="unicode_escape")
    row_cond = trials_df.loc[trials_df['_condition'] == patient_condition_id]
    cond_id = row_cond['_kind'].values[0]
    # print('BLA ConditionID: ', cond_id)
    return cond_id


def get_therapy_recommendation(patient_id, patient_condition, dataset):

    # TODO recommendation based on HYBRID: ITEM-ITEM COLLABORATIVE FILTERING and Global Baseline Estimate
    # get Global recommendation
    global_therapies_recommendation_for_conditions_by_success(dataset, patient_condition)

    # create_patients_therapies_similarity_by_success(dataset, patient_id)
    # get_patient_therapies_ranking
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
    # create_patients_therapies_similarity_by_success(dataset)    # top successful therapies for a patient

    # TODO cases
    # TODO recommendation based on USER-USER COLLABORATIVE FILTERING
    # create_patients_similarity_by_condition(dataset)

    # TODO recommendation based on ITEM-ITEM COLLABORATIVE FILTERING
    # create_conditions_similarity_by_patients(dataset)
    # create_therapies_similarity_by_patients(dataset)

    # TODO ALGO CS
    # filtered_df = filter_df_features(dataset)

    # TODO retreat
    # filtered_df = filter_df_features(dataset)
    # display(filtered_df.head())
    # display(filtered_df.describe())
    #
    # # creating mean success_rate for treatments
    # success_rate = pd.DataFrame(filtered_df.groupby('_therapy')['_successful'].mean())
    # display(success_rate.head())
    #
    # # creating number_of_successful_trials
    # success_rate['number_of_successful_trials'] = filtered_df.groupby('_therapy')['_successful'].count()
    # display(success_rate.head())

    # # plotting the jointplot
    # h = sns.jointplot(x='_successful', y='number_of_successful_trials', data=success_rate)
    #
    # # JointGrid has a convenience function
    # h.set_axis_labels('x', 'y', fontsize=16)
    #
    # # or set labels via the axes objects
    # h.ax_joint.set_xlabel('success_rate', fontweight='bold')
    #
    # # or set labels via the axes objects
    # h.ax_joint.set_ylabel('number_of_trials', fontweight='bold')
    #
    # # also possible to manipulate the histogram plots this way, e.g.
    # h.ax_marg_y.grid('on')  # with ugly consequences...
    #
    # # labels appear outside of plot area, so auto-adjust
    # h.figure.tight_layout()
    #
    # plt.show()

    # creating Condition-Therapy Interaction Matrix
    # therapy_matrix_CTI = filtered_df.pivot_table(index='_kind', columns='_therapy', values='_successful')
    # display(therapy_matrix_CTI.head())
    #
    # # most successful therapies
    # display(success_rate.sort_values('number_of_successful_trials', ascending=False).head())


def main():
    # input Dataset, patient and condition

    # dataset = 'datasetB.json'
    # cases = "datasetB_cases.txt"

    dataset = 'dataset_shaker.json'
    cases = "dataset_shaker_cases.txt"

    # dataset = 'dataset_sample.json'
    # cases = "dataset_sample_cases.txt"

    # input

    # print Dataset summary
    print_dataset_summary(dataset)
    create_patients_trials_dataframe(dataset)

    with open(cases, newline='') as file_in:
        next(file_in)
        for row in file_in:
            row = row.strip()
            row = row.split('\t\t')

            patient_id = row[0]
            patient_condition = row[1]

            print('PatientID: ', patient_id)
            print('Patient_condition: ', patient_condition)

            try:
                original_condition_id = get_condition_id(dataset, patient_condition)
                print('Condition: ', original_condition_id, '\n')

                get_therapy_recommendation(patient_id, original_condition_id, dataset)
            except:
                print('Condition NOT FOUND. Retrieval failed from the Dataset.')

            print()
            # break


if __name__ == '__main__':
    main()