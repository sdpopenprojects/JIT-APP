import numpy as np
import pandas as pd
import time

from datetime import datetime


def union_data(data, commit_date):
    # replace NaN and infinite values with 0
    data = data.replace([np.nan, np.inf, -np.inf], 0)
    # bool to 0 1
    data = data.fillna(0)

    # change format
    # if commit_date not in data.columns:
    #     print('data does exists a column commitdate')

    # commit_date = 'date'
    # data[commit_date] = (pd.to_datetime(data[commit_date], unit='s', origin=pd.Timestamp('1970-01-01')).dt.strftime('%Y-%m'))

    # commit_date = 'commitTime'
    # data[commit_date] = (pd.to_datetime(data[commit_date]).dt.strftime('%Y-%m'))

    data[commit_date] = [datetime.strftime(date_str, '%Y-%m') for date_str in pd.to_datetime(data[commit_date])]

    data = data.sort_values(commit_date)
    
    # ordered data by commit time
    temp = data[commit_date].unique()
    temp.sort()
    total_folds = len(temp)
    sub = [None] * total_folds
    for fold in range(total_folds):
        sub[fold] = data[data[commit_date] == temp[fold]]
    return total_folds, sub


# time-wise cross validation
def time_wise_CV(data, gap, commit_date):
    # divide the data of same month into same fold
    total_folds, sub = union_data(data, commit_date)
    train_folds = []
    gap_folds = []
    test_folds = []
    for fold in range(total_folds):
        if fold + gap * 3 > total_folds:
            continue
        trn_fold = pd.concat([sub[fold + i].iloc[:, 1:] for i in range(gap)])  # train set
        gap_fold = pd.concat([sub[fold + gap + i].iloc[:, 1:] for i in range(gap)])  # gap
        te_fold = pd.concat([sub[fold + gap * 2 + i].iloc[:, 1:] for i in range(gap)])  # test set

        train_folds.append(trn_fold)
        gap_folds.append(gap_fold)
        test_folds.append(te_fold)

    return train_folds, test_folds, gap_folds
