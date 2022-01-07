# Description: Build a Therapy Recommendation System
# Author: Shaker Mahmud Khandaker
# Course: Data Mining
# Matricola: 229221


# imports
import json
import random
from datetime import datetime
import pandas as pd

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

    df.to_csv(r'DF_Conditions_sample.csv', index=None)


def create_therapies_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    df = pd.json_normalize(data['Therapies'])

    df.to_csv(r'DF_Therapies_sample.csv', index=None)


def create_patients_dataframe(filename):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # trials
    df_trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='_')

    # df_trials.sort_values('_id')
    # df_trials.to_csv(r'DF_Trials_sample.csv', index=None)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "condition", "id", errors='ignore', record_prefix='_')

    # df_conditions.sort_values('_id')
    # df_conditions.to_csv(r'DF_Conditions_sample.csv', index=None)

    print(df_trials)
    print()
    print()
    print(df_conditions)

    merged_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])
    # merged_df.sort_values('_id')
    # merged_df.to_csv(r'DF_Patients_sample.csv', index=None)

    print()
    print()
    print(merged_df)
    # merged_df.sort_values('_id')
    merged_df.to_csv(r'DF_Patients_sample.csv', index=None)


# get unique file name
def get_unique_file_name(prefix, suffix):
    # return 'dataframe_sample_' + get_current_datetime_string() + '.csv'
    return prefix + get_current_datetime_string() + suffix


def main():
    # print(get_unique_file_name('dataframe_sample_', '.csv'))

    # print file summary
    print_dataset_summary(dataset)
    create_conditions_dataframe(dataset)
    create_therapies_dataframe(dataset)
    create_patients_dataframe(dataset)


if __name__ == '__main__':
    main()
