# Data Mining

import json
import traceback
import math
import numpy as np
import pandas as pd

print(
    '# Description: A Therapy Recommendation System\n# Author: Shaker Mahmud Khandaker\n# '
    'Course: Data Mining\n# Matricola: 229221\n')


def get_patient_top_cond_therapy_pair(dataframe, matrix, pid):
    top_cond_therapy_pair = calculate_top_therapies(matrix, 5)
    print("Top Condition-Therapy Pair:\n")
    print(top_cond_therapy_pair)

    dataframe = dataframe.groupby('CT Pair')['sub_successful'].mean()

    tops = top_cond_therapy_pair.loc[int(pid), :]
    # print('Top Condition-Therapy Pair for Patient ' + pid + ': ')
    df_ranked = pd.DataFrame(columns=['Condition-Therapy', 'Success'])

    for items in tops:
        try:
            df_ranked = df_ranked.append({'Condition-Therapy': items, 'Success': dataframe[items]}, ignore_index=True)
            print(dataframe[items])
        except:
            break
    # print(df_ranked['Condition-Therapy'][0])

    return df_ranked


def global_therapies_recommendation_for_conditions_by_success(dataset, condition, count):
    with open(dataset, encoding="utf8") as f:
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
    similar_neighbours = calculate_top_therapies(condition_therapy_matrix, 5)
    print(similar_neighbours.head(5))

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


def standardize(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row


def get_similar_cond_therapy_pair(cond_therapy_pair, success_rate, item_similarity_df):
    similar_score = item_similarity_df[cond_therapy_pair] * (success_rate - 50)
    similar_score = similar_score.sort_values(ascending=False)
    print(similar_score)
    return similar_score


def get_condition_id(filename, patient_condition_id):
    with open(filename, encoding="unicode_escape") as f:
        data = json.load(f)

    # conditions
    df_conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='sub_')
    # print(df_conditions)
    row_cond = df_conditions.loc[df_conditions['sub_id'] == patient_condition_id]
    cond_id = row_cond['sub_kind'].values[0]
    return cond_id


def calculate_top_therapies(dataframe, n):
    sim = np.argsort(dataframe.values, axis=1)[:, :n]

    dataframe = dataframe.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index,
                                                    index=['rank_{}'.format(i) for i in range(1, n + 1)]), axis=1)
    return dataframe


def get_prediction(ct_pair_matrix, ratings_ct, condid, ct_pair, success):
    # Fetching CT_pair for all patent with successrate
    comparison_CT_pair = ct_pair_matrix[ct_pair]
    # print(comparison_CT_pair.head(5))

    # Finding the correlation to with different CT_pair
    similar_to_CT_pair = ct_pair_matrix.corrwith(comparison_CT_pair)
    # print(similar_to_CT_pair.head(5))

    # Creating a dataframe to bring in #of therapy
    corr_ct_pair = pd.DataFrame(similar_to_CT_pair, columns=['correlation'])
    corr_ct_pair.dropna(inplace=True)
    # print(corr_ct_pair.head(5))

    # Bringing the success counts
    corr_ct_pair = corr_ct_pair.join(ratings_ct['number_of_CT_Pair'])
    # print(corr_ct_pair.head(5))

    # filtering out the therapies based on maximum applied
    suggested_ct_pair_df = corr_ct_pair[corr_ct_pair['number_of_CT_Pair'] > -1].sort_values(
        by='correlation', ascending=False)
    suggested_ct_pair_df = suggested_ct_pair_df.iloc[1:]
    suggested_ct_pair_df['CT_pair'] = suggested_ct_pair_df.index

    suggested_ct_pair_df.columns = ['Correlation', 'count', 'CT_pair']
    # print('suggested_ct_pair_df.columns')
    # print(suggested_ct_pair_df)

    suggested_ct_pair_lists = []

    try:
        suggested_ct_pair_df[['Condition', 'Therapy']] = suggested_ct_pair_df.CT_pair.str.split('_',
                                                                                                1).tolist()
        # print('suggested_ct_pair_df BEFORE FILTERING WITH COND: ', len(suggested_ct_pair_df))
        # print(suggested_ct_pair_df)

        suggested_ct_pair_df = suggested_ct_pair_df[suggested_ct_pair_df['Condition'] == condid]
        suggested_ct_pair_df['Success'] = suggested_ct_pair_df['Correlation']*success
        # print(suggested_ct_pair_df)
        # print('suggested_ct_pair_df AFTER FILTERING WITH COND: ', len(suggested_ct_pair_df))
        # print(suggested_ct_pair_df)
        columns = ['Correlation', 'Therapy', 'Success']
        suggested_ct_pair_df = suggested_ct_pair_df[columns]
        suggested_ct_pair_lists = suggested_ct_pair_df.values.tolist()

    except:
        print()

    return suggested_ct_pair_lists


def prepare_outputs(data, prediction_df, pid, pcond):
    # Adding Therapy names to suggestions
    therapies = pd.json_normalize(data['Therapies'])
    suggested_therapy_df = pd.merge(prediction_df, therapies, how='left', left_on='Therapy',
                                    right_on='id')

    # Filtering other fields
    # cols = ['id', 'name', 'Correlation', 'Success']
    cols = ['id', 'name']
    suggested_therapy_df = suggested_therapy_df[cols]
    # Writing to file
    suggested_therapy_df.head(5).to_csv(r'result_' + pid + '_' + pcond + '.csv',
                                        index=None)
    print(suggested_therapy_df.head(5))
    return 0


def calculate_rmse(rmse_sample, actual, predicted):

    rmse_trials_df = pd.read_csv(rmse_sample, encoding="unicode_escape")
    MSE = np.square(np.subtract(actual, predicted)).mean()
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:\n")
    print(RMSE)
    RMSE.to_csv(r'rmse_output.csv', index=None)


def main():
    dataset = 'datasetB.json'
    test = "datasetB_cases.txt"
    rmse_sample = "rmse_input.csv"
    # dataset = 'datasetB_sample.json'
    # test = "datasetB_cases_sample.txt"

    with open(dataset, encoding="utf8") as f:
        data = json.load(f)

    trials = pd.json_normalize(data['Patients'], "trials", ["id", "name"], errors='ignore', record_prefix='sub_')

    conditions = pd.json_normalize(data['Patients'], "conditions", "id", errors='ignore', record_prefix='sub_')

    trails_dataframe = pd.merge(trials, conditions, how='left', left_on=['sub_condition', 'id'],
                                right_on=['sub_id', 'id'])
    print(trails_dataframe)
    print("Dataset Summary")
    print(trails_dataframe.describe())

    # FILTERING OUT NOT CURED TRIALS
    print("Filtering out not cured trials:\n")
    trails_dataframe = trails_dataframe[trails_dataframe['sub_isCured'] == True]
    print(trails_dataframe)

    print("Filtered Dataset Summary")
    print(trails_dataframe.describe())

    # CREATING CONDITION-THERAPY PAIR COLUMN: MULTIPLYING CONDITION WITH THERAPY
    trails_dataframe["CT Pair"] = trails_dataframe["sub_kind"] + '_' + trails_dataframe["sub_therapy"]

    # sub_condition
    # sub_end
    # sub_id_x
    # sub_start
    # sub_successful
    # sub_therapy
    # id
    # name
    # sub_cured
    # sub_diagnosed
    # sub_id_y
    # sub_isCured
    # sub_isTreated
    # sub_kind

    # FILTERING OTHER FIELDS
    feature_cols = ['id', 'CT Pair', 'sub_successful', 'sub_therapy', 'sub_kind']
    trails_dataframe = trails_dataframe[feature_cols]

    print("FINAL DATAFRAME:\n")
    print(trails_dataframe)
    # trails_dataframe.to_csv(r'FINAL_df.csv', index=None)

    # FINAL DATAFRAME COLUMNS: id, CT Pair, sub_successful

    # Creating mean of success rate for all CT Pair
    ratings_cond_therapy = pd.DataFrame(trails_dataframe.groupby('CT Pair')['sub_successful'].mean())
    print("Creating mean of success rate for all CT Pair")
    print(ratings_cond_therapy.head())

    # Creating number of CT Pair data
    ratings_cond_therapy['number_of_CT_Pair'] = trails_dataframe.groupby('CT Pair')['sub_successful'].count()
    print("Creating number of CT Pair data")
    print(ratings_cond_therapy.head(5))

    print('Patient-CondTherapy interaction matrix: \n')
    # Creating Patient-CondTherapy interaction matrix
    cond_therapy_pair_matrix = trails_dataframe.pivot_table(index='id', columns='CT Pair', values='sub_successful')
    print(cond_therapy_pair_matrix)

    # TODO for Global Baseline
    print('Most Applied Therapy: \n')
    # Most Applied therapy
    print(ratings_cond_therapy.sort_values('number_of_CT_Pair', ascending=False).head(5))
    #
    print('Most Successful Therapy: \n')
    # Most Successful CONDITION-THERAPY PAIR
    print(ratings_cond_therapy.sort_values('sub_successful', ascending=False).head(5))

    ratings_cond_therapy['CT_pair'] = ratings_cond_therapy.index

    with open(test, newline='') as file_in:
        next(file_in)
        for row in file_in:
            row = row.strip()
            pid = row.split(None, 1)[0]
            pcond = row.split(None, 1)[1]
            condid = get_condition_id(dataset, pcond)
            print('Condition: ', condid, '\n')

            # GET Patient Trials
            patient_dataframe = trails_dataframe[trails_dataframe['id'] == int(pid)]
            print("Patient Trials DF")
            print(patient_dataframe)

            predictions = []
            # Iterating through Patient Trials and Generating Prediction based on each trials
            for therapy, success in zip(patient_dataframe['sub_therapy'], patient_dataframe['sub_successful']):
                # print(therapy, success)
                try:
                    # Create CT pair and get prediction
                    patient_ct_pair = condid + '_' + therapy
                    print('CT Pair:', patient_ct_pair, '    ', success)

                    # global success for ct_pair
                    ct_gb_row = ratings_cond_therapy[ratings_cond_therapy['CT_pair'] == patient_ct_pair]
                    ct_gb_success = ct_gb_row.iat[0, 0]
                    print('Global success for CT Pair: ', ct_gb_success)
                    # print('DONE')

                    local_prediction = get_prediction(cond_therapy_pair_matrix, ratings_cond_therapy, condid,
                                                      patient_ct_pair, ct_gb_success)

                    for item in local_prediction:
                        # Append predictions
                        predictions.append(item)

                except Exception:
                    traceback.print_exc()
                    print('Entry not found.')

            # print('All Predictions:\n')
            # predictions = list(filter(None, predictions))

            prediction_df = pd.DataFrame(predictions, columns=['Correlation', 'Therapy', 'Success'])
            # prediction_df = pd.DataFrame()
            # for item in predictions:
            #

            prediction_df = prediction_df.sort_values(by='Success', ascending=False)
            print('Predictions:')
            print(prediction_df)
            # break
            # calculate rmse
            # calculate_rmse(rmse_sample, prediction_df)
            # prepare output
            prepare_outputs(data, prediction_df, pid, pcond)


if __name__ == '__main__':
    main()
