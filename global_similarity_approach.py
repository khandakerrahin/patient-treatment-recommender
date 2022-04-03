# Description: Build a Therapy Recommendation System
# Author: Shaker Mahmud Khandaker
# Course: Data Mining
# Matricola: 229221


# imports
import json
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from IPython.display import display

line = "------------------------------------------------------"
print(line)
print(
    '# Description: A Therapy Recommendation System\n# Author: Shaker Mahmud Khandaker\n# '
    'Course: Data Mining\n# Matricola: 229221')

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

    global therapy_count, condition_count, patient_count, trial_count, line

    condition_count = len(data['Conditions'])
    therapy_count = len(data['Therapies'])
    patient_count = len(data['Patients'])
    trial_count = 0

    for patient in data['Patients']:
        # print('Trials: ', len(patient['trials']))
        trial_count += len(patient['trials'])

    # print dataset summary
    print(line)
    print('Dataset Summary:\n')
    print('# Conditions: ', condition_count)
    print('# Therapies: ', therapy_count)
    print('# Patients: ', patient_count)
    print('# Trials: ', trial_count)
    print(line)


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

    # df.to_csv(r'DF_Therapies_temp.csv', index=None)
    return df


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

    # similar_neighbours.to_csv(r'similar_patients_by_therapy.csv')


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

    # similar_neighbours.to_csv(r'similar_conditions_by_patients.csv')


def create_therapies_similarity_by_patients(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # therapy
    df_therapy = pd.json_normalize(data['Patients'], "trials", "id", errors='ignore', record_prefix='_')

    df_therapy = df_therapy.groupby(['id', '_therapy']).size().unstack(fill_value=0)
    print(df_therapy)

    similarity_df = df_therapy.corr(method='pearson')
    display(similarity_df.head(10))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    # similar_neighbours.to_csv(r'similar_therapies_by_patients.csv')


def get_patient_therapies_ranking(filename, patient_id):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # therapy
    df_therapy = pd.json_normalize(data['Patients'], "trials", "id", errors='ignore', record_prefix='_')
    # print(df_therapy)
    patient_therapy_matrix = df_therapy.pivot_table(index='id', columns='_therapy', values='_successful')

    # patient_therapy_matrix = patient_therapy_matrix.fillna(0)
    # patient_therapy_matrix = patient_therapy_matrix.fillna(condition_therapy_matrix.mean(axis=0))
    # display(patient_therapy_matrix.head(20))

    # patient_therapy_matrix.to_csv(r'patient_therapy_matrix.csv')

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(patient_therapy_matrix, therapy_count)
    # display(similar_neighbours.head(20))
    # print('HERE', patient_id)
    # similar_neighbours.to_csv(r'patient_therapy_similarity.csv')
    df_therapy = df_therapy[df_therapy['id'] == int(patient_id)]
    # print(df_therapy)
    df_therapy = df_therapy.groupby('_therapy')['_successful'].mean()

    # print('FILTERED AND GROUPED FOR '+condition)
    # print(df_therapy)

    rowData = similar_neighbours.loc[int(patient_id), :]
    print('Top Successful Therapies for USER:' + patient_id)

    df_ranked = pd.DataFrame(columns=['Therapy', 'Success'])

    print('PATIENT RANKING: \n')
    for items in rowData:
        try:
            df_ranked = df_ranked.append({'Therapy': items, 'Success': df_therapy[items]}, ignore_index=True)
        except:
            break
    print(df_ranked)
    # dfg = df_ranked.describe()

    # print(dfg)
    # dfg.to_csv(r'BASSBSSADDDDDDDDEEE.csv')


def global_therapies_recommendation_for_conditions_by_success(filename, condition, count):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)

    # trials
    df_trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='_')

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')

    # print(df_trials)
    # print(df_conditions)

    trials_df = pd.merge(df_trials, df_conditions, how='left', left_on=['_condition', 'id'], right_on=['_id', 'id'])

    condition_therapy_matrix = trials_df.pivot_table(index='_kind', columns='_therapy', values='_successful')

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(condition_therapy_matrix, 5)
    display(similar_neighbours.head(5))

    trials_df = trials_df[trials_df['_kind'] == condition]

    trials_df = trials_df.groupby('_therapy')['_successful'].mean()

    # print('FILTERED AND GROUPED FOR '+condition)
    # display(trials_df.head(5))

    rowData = similar_neighbours.loc[condition, :]
    print('Top Recommended Therapies for ' + condition + ': ')
    df_ranked = pd.DataFrame(columns=['Therapy', 'Success'])

    for items in rowData:
        try:
            df_ranked = df_ranked.append({'Therapy': items, 'Success': trials_df[items]}, ignore_index=True)
        except:
            break
    print(df_ranked['Therapy'][0])

    return df_ranked


def create_therapies_similarity_by_therapies(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    df_conditions = pd.json_normalize(data['Therapies'])

    print(df_conditions)

    df_conditions = df_conditions.groupby(['type', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))
    # similarity_df.to_csv(r'similar_therapies_scores.csv')

    # print(check_if_below_average(similarity_df, 0.24))

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    # similar_neighbours.to_csv(r'similar_therapies_by_therapies.csv')


def create_conditions_similarity_by_conditions(filename):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    df_conditions = pd.json_normalize(data['Conditions'])

    print(df_conditions)

    df_conditions = df_conditions.groupby(['type', 'id']).size().unstack(fill_value=0)
    # print(df_conditions)

    similarity_df = df_conditions.corr(method='pearson')
    display(similarity_df.head(10))
    # similarity_df.to_csv(r'similar_conditions_scores.csv')

    # top 5 neighbours
    similar_neighbours = find_n_neighbours(similarity_df, 5)
    display(similar_neighbours.head(20))

    # similar_neighbours.to_csv(r'similar_conditions_by_conditions.csv')


def get_min_df(dataframe, index):
    return dataframe.iloc[index].min()


def get_max_df(dataframe, index):
    return dataframe.iloc[index].max()


def check_if_below_average(dataframe, value):
    min_value = get_min_df(dataframe, 0)
    max_value = get_max_df(dataframe, 0)

    print('Min: ', min_value)
    print('Max: ', max_value)

    average_value = (min_value + max_value) / 2

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

    return merged_df[columns]


def get_useful_features(data):
    useful_features = []

    for c in range(0, data.shape[0]):
        useful_features.append(
            data['id'][c] + ' ' + data['_kind'][c] + ' ' + data['_therapy'][c] + ' ' + data['_successful'][c])

    return useful_features


def get_condition_id(filename, patient_condition_id):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='_')
    # print(df_conditions)
    row_cond = df_conditions.loc[df_conditions['_id'] == patient_condition_id]
    cond_id = row_cond['_kind'].values[0]
    return cond_id


def get_therapy_recommendation(patient_id, condition_id, patient_condition, dataset):
    # TODO recommendation based on HYBRID: ITEM-ITEM COLLABORATIVE FILTERING and Global Baseline Estimate
    # get Global recommendation
    global_top_therapy_for_condition = global_therapies_recommendation_for_conditions_by_success(dataset, condition_id, 1)

    # USER-ITEM COLLABORATIVE FILTERING
    filtered_df = filter_df_features(dataset)
    display(filtered_df.head())

    print("Merging Condition and Therapy\n")

    # filtered_df["Cond*Therapy"] = filtered_df["_kind"] + '_' + filtered_df["_therapy"]
    # # display(filtered_df.describe())
    #
    # display(filtered_df.head())

    # creating mean success_rate for treatments
    success_rate = pd.DataFrame(filtered_df.groupby('_therapy')['_successful'].mean())
    display(success_rate.head(5))

    # creating number_of_successful_trials
    success_rate['number_of_successful_trials'] = filtered_df.groupby('_therapy')['_successful'].count()
    display(success_rate.head(5))

    # creating User-Therapy Interaction Matrix
    therapy_matrix_CTI = filtered_df.pivot_table(index='id', columns='_therapy', values='_successful')
    display(therapy_matrix_CTI.head(5))

    # most successful therapies
    # display(success_rate.sort_values('number_of_successful_trials', ascending=False).head())
    threshold = success_rate.sort_values('number_of_successful_trials', ascending=False).head()['number_of_successful_trials'][2]

    threshold = threshold - threshold * 25 / 100
    # print(threshold)

    # Making Recommendation for a specific Famous Therapy IE: Th33
    specific_therapy_patient_success_rate = therapy_matrix_CTI[global_top_therapy_for_condition['Therapy'][0]]
    display(specific_therapy_patient_success_rate.head(5))

    # finding the correlation with different Therapies
    similar_to_specific_therapy = therapy_matrix_CTI.corrwith(specific_therapy_patient_success_rate)
    display(similar_to_specific_therapy.head(5))

    # create a threshold for minimum number of trials
    # creating a dataframe to bring in #of trials
    corr_specific_therapy = pd.DataFrame(similar_to_specific_therapy, columns=['Correlation'])
    corr_specific_therapy.dropna(inplace=True)

    # adding success_rate
    corr_specific_therapy = corr_specific_therapy.join(success_rate['number_of_successful_trials'])
    display(corr_specific_therapy.head(5))

    # filtering out by selecting most applied therapies
    recommended_therapies_df = corr_specific_therapy[corr_specific_therapy['number_of_successful_trials'] > threshold].sort_values(by='Correlation', ascending=False).head(10)
    # display(recommended_therapies_df.head(5))
    df_therapies = create_therapies_dataframe(dataset)
    # display(df_therapies.head(5))

    recommended_therapies_df = pd.merge(recommended_therapies_df, df_therapies, how='left', left_on='_therapy', right_on='id')

    columns = ['id', 'name', 'Correlation']

    recommended_therapies_df = recommended_therapies_df[columns]
    recommended_therapies_df = recommended_therapies_df.iloc[1:]
    display(recommended_therapies_df.head(5))

    recommended_therapies_df.head(5).to_csv(r'result_' + patient_id + '_' + patient_condition + '.csv', index=None)


def main():
    # input Dataset, patient and condition

    # dataset = 'datasetB.json'
    # cases = "datasetB_cases.txt"

    dataset = 'datasetB_sample.json'
    cases = "datasetB_cases_sample.txt"

    # print Dataset summary
    print_dataset_summary(dataset)
    # create_patients_trials_dataframe(dataset)

    # input test cases
    with open(cases, newline='') as file_in:
        next(file_in)
        for row in file_in:
            row = row.strip()
            patient_id = row.split(None, 1)[0]
            patient_condition = row.split(None, 1)[1]

            print(line)
            print('PatientID: ', patient_id)
            print('Patient_condition: ', patient_condition)

            try:
                original_condition_id = get_condition_id(dataset, patient_condition)
                print('Condition: ', original_condition_id, '\n')

                # get_patient_therapies_ranking(dataset, patient_id)

                get_therapy_recommendation(patient_id, original_condition_id, patient_condition, dataset)
            except Exception:
                print(traceback.format_exc())
                print('Condition NOT FOUND. Retrieval failed from the Dataset.')

            print(line)
            # break


if __name__ == '__main__':
    main()
