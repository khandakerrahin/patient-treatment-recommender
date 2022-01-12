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


def create_patients_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # trials
    df_trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='_')

    # df_trials.sort_values('_id')
    # df_trials.to_csv(r'DF_Trials_temp.csv', index=None)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    # df_conditions.sort_values('_id')
    # df_conditions.to_csv(r'DF_Conditions_temp.csv', index=None)

    # print(df_trials)
    # print()
    # print()
    # print(df_conditions)

    merged_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])
    # merged_df.sort_values('_id')
    # merged_df.to_csv(r'DF_Patients_temp.csv', index=None)

    # print()
    # print()
    # print(merged_df)
    # merged_df.sort_values('_id')
    merged_df.to_csv(r'DF_Patients_temp.csv', index=None)


# def get_user_similar_conditions( user1, user2 ):
#     common_movies = Rating_avg[Rating_avg.userId == user1].merge(
#     Rating_avg[Rating_avg.userId == user2],
#     on = "movieId",
#     how = "inner" )
#     return common_movies.merge( movies, on = 'movieId' )


def create_patients_condition_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    # df_conditions.sort_values('_id')
    # df_conditions.to_csv(r'DF_Conditions_temp.csv', index=None)

    # print(df_conditions)
    df_conditions = df_conditions.groupby(['_kind', 'id']).size().unstack(fill_value=0)

    # print(df_conditions)

    # df_conditions.to_csv(r'condition_user_matrix.csv')

    user_similarity_df = df_conditions.corr(method='pearson')
    display(user_similarity_df.head(10))

    # a = get_user_similar_conditions(370, 86309)
    # a = a.loc[:, ['rating_x_x', 'rating_x_y', 'Condition']]
    # a.head()

    # top 30 neighbours for each user
    sim_user_30_u = find_n_neighbours(user_similarity_df, 5)
    display(sim_user_30_u.head(20))

    sim_user_30_u.to_csv(r'sim_user_30_u.csv')

    # merged_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])
    # merged_df.sort_values('_id')
    # merged_df.to_csv(r'DF_Patients_temp.csv', index=None)

    # merged_df.to_csv(r'DF_Patients_temp.csv', index=None)


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
    # print(get_unique_file_name('dataframe_sample_', '.csv'))

    # print file summary
    print_dataset_summary(dataset)
    create_conditions_dataframe(dataset)
    create_therapies_dataframe(dataset)
    create_patients_dataframe(dataset)
    create_patients_condition_dataframe(dataset)

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


if __name__ == '__main__':
    main()
