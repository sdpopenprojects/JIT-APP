import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from utilities.cliffsDelta import cliffsDelta


def get_files(file_path, model_name):
    temp_files = [f for f in sorted(os.listdir(file_path + model_name + '/')) if f.endswith('.csv')]
    return temp_files


if __name__ == '__main__':

    path = r"../result_compare/"
    clf = 'NB'
    model_name = clf + '_method'
    model_name_base1 = clf + '_change'

    measurename = ['precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC',
                   'popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA']

    # files = [f for f in sorted(os.listdir(path + model_name + '/')) if f.endswith('.csv')]
    current_files = get_files(path, model_name)
    files_num = len(current_files)

    base_files = get_files(path, model_name_base1)

    files_list = []
    d_list = np.zeros([files_num, len(measurename)])
    size_list = []
    for i in range(files_num):
        current_file = current_files[i]
        file_name = current_file[:-4]
        files_list.append(file_name)

        current_file_path = os.path.join(path + model_name + '/', current_file)
        current_df = pd.read_csv(current_file_path, header=None)
        current_df.columns = measurename

        # baseline
        base_file = base_files[i]
        base_file_path = os.path.join(path + model_name_base1 + '/', base_file)
        base_df = pd.read_csv(base_file_path, header=None)
        base_df.columns = measurename

        # precision, recall, pf, f_measure, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall, Eprecision, Efmeasure,
        # PMI, IFA
        temp_size = []
        for j in range(len(measurename)):
            mea = measurename[j]
            if mea in ['pf', 'cPMI', 'cIFA']:
                d, size = cliffsDelta(-current_df[mea], -base_df[mea])
            else:
                d, size = cliffsDelta(current_df[mea], base_df[mea])
                # d, size = cliffsDelta(np.array(current_df[mea]), np.array(base_df[mea]))
            d_list[i, j] = d
            temp_size.append(size)

        size_list.append(temp_size)

    d_list = DataFrame(d_list)
    d_list.columns = measurename
    d_list.index = files_list
    d_list.to_csv(path + clf + '_cliff_d.csv')

    size_list = DataFrame(size_list)
    size_list.columns = measurename
    size_list.index = files_list
    size_list.to_csv(path + clf + '_cliff_size.csv')
