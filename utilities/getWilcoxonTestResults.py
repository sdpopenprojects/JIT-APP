import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import wilcoxon, mannwhitneyu, ranksums


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
    pvalue_list = np.zeros([files_num, len(measurename)])
    corrected_pvalue_list = np.zeros([files_num, len(measurename)])
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

            if mea in ['pf','cPMI', 'cIFA']:
                # Wilcoxon signed-rank test
                # p = wilcoxon(-current_df[mea], -base_df[mea])
                # Wilcoxon rank-sum test
                p = ranksums(-current_df[mea], -base_df[mea])
            else:
                # p = wilcoxon(current_df[mea], base_df[mea])
                p = ranksums(current_df[mea], base_df[mea])

            pvalue_list[i, j] = p.pvalue

            # use Bonferroni correction
            # # number of groups
            # k = 2
            # # number of observations
            # n = len(current_df[mea]) + len(base_df[mea])
            #
            # corrected_pvalue = min(p.pvalue*k*n, 1.0)
            # corrected_pvalue_list[i, j] = corrected_pvalue

            if p.pvalue < 0.05:
                if mea in ['pf', 'cPMI', 'cIFA']:
                    if current_df[mea].median() < base_df[mea].median():
                        temp_size.append('win')
                    else:
                        temp_size.append('lose')
                else:
                    if current_df[mea].median() > base_df[mea].median():
                        temp_size.append('win')
                    else:
                        temp_size.append('lose')
            else:
                temp_size.append('tie')
        size_list.append(temp_size)

    # save to csv file
    # pvalue_list = DataFrame(pvalue_list)
    # pvalue_list.columns = measurename
    # pvalue_list.index = files_list
    # pvalue_list.to_csv(path + clf + '_pvalue.csv')
    #
    # corrected_pvalue_list = DataFrame(corrected_pvalue_list)
    # corrected_pvalue_list.columns = measurename
    # corrected_pvalue_list.index = files_list
    # corrected_pvalue_list.to_csv(path + clf + '_corrected_pvalue.csv')

    size_list = DataFrame(size_list)
    size_list.columns = measurename
    size_list.index = files_list
    size_list.to_csv(path + clf + '_pvalue_size.csv')
