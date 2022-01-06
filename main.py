# Description: Build a Therapy Recommendation System
# Author: Shaker Mahmud Khandaker
# Course: Data Mining
# Matricola: 229221


# imports
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd


print(
    '# Description: Build a Therapy Recommendation System\n# Author: Shaker Mahmud Khandaker\n# '
    'Course: Data Mining\n# Matricola: 229221\n')


def get_current_datetime_string():
    # datetime object containing current date and time
    now = datetime.now()

    # print("now =", now)

    # ddmmYY_HMS
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    # print("date and time =", dt_string)

    return dt_string


# main
# read DATASET file
with open('dataset_shaker.json', encoding="utf8") as f:
    data = json.load(f)

print('Dataset Summary:\n')
print('# Conditions: ', len(data['Conditions']))
print('# Therapies: ', len(data['Therapies']))
print('# Patients: ', len(data['Patients']))
# print('Trials: ', len(data['Patients'][2]['trials']))

trialCounts = 0

for patient in data['Patients']:
    # print('Trials: ', len(patient['trials']))
    trialCounts += len(patient['trials'])

print('# Trials: ', trialCounts)

print('result_' + get_current_datetime_string())

print()
