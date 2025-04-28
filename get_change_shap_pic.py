import os
import pickle
import warnings
import numpy as np
import pandas as pd

import shap
import matplotlib.pyplot as plt



from utilities.AutoSpearman import AutoSpearman

from imblearn.over_sampling import SMOTE


def get_all_features_name(save_path,pro_num):
    all_feature_names = set()
    for i in range(pro_num):

        project_name = project_names[i]
        project_name = project_name[:-4]

        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "rb") as f:
            shap_values_loaded = pickle.load(f)

        if CLF == 'RF' or CLF == 'DT':
            shap_values_loaded = shap_values_loaded[:, :, 1]
        all_feature_names.update(shap_values_loaded.feature_names)
    all_feature_names = sorted(list(all_feature_names))
    return all_feature_names

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'./result_change/RF/'

    CLF = 'RF' # 'NB','DT','GBM','KNN','LR','RF'
    # model_name = 'LR_method'

    project_names = sorted(os.listdir('./data/change/'))
    path = os.path.abspath('./data/change/')
    pro_num = len(project_names)

    column_name = ['commit_date', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp',
                   'rexp', 'sexp', 'buggy_B2']
    commit_date = column_name[0]
    # gap = 2

    churn_name = ['commit_date', 'LA', 'LD', 'LT', 'bug']
    all_feature_names = get_all_features_name(save_path,pro_num)
    all_shap_df_list = []
    all_X_df_list = []
    for i in range(0, pro_num):
        project_name = project_names[i]
        print("doing "+ project_name)
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)

        # 按提交时间顺序排序
        data = data.sort_values(by=['commit_date'])

        project_name = project_name[:-4]

        data = data[column_name]
        data.rename(columns={'buggy_B2': 'bug'}, inplace=True)

        data['bug'][data['bug'] == 'clean'] = 0
        data['bug'][data['bug'] == 'buggy'] = 1


        ori_data = data.drop(['commit_date', 'bug'], axis=1)
        data['effort'] = data['la'] + data['ld']

        # feature selection
        sel_data = AutoSpearman(ori_data)

        new_data = pd.concat([data['commit_date'], sel_data, data[['effort', 'bug']]], axis=1)

        train_data, val_data, test_data = np.split(new_data, [int(0.6 * len(new_data)), int(0.8 * len(new_data))])

        train_label = train_data['bug']
        val_label = val_data['bug']
        test_label = test_data['bug']

        train_label = train_label.astype(int)
        val_label = val_label.astype(int)
        test_label = test_label.astype(int)

        train_data = train_data.drop(['commit_date', 'bug'], axis=1)
        val_data = val_data.drop(['commit_date', 'bug'], axis=1)
        test_data = test_data.drop(['commit_date', 'bug'], axis=1)

        # only one class
        if len(np.unique(train_label)) < 2 or len(np.unique(test_label)) < 2 or sum(train_label) < 2 \
                or len(train_label) - sum(train_label) < 2:
            continue

        LOC = test_data['effort']

        # keep fix unchanged
        if 'fix' in train_data.columns:
            train_fix = train_data['fix']
            val_fix = val_data['fix']
            test_fix = test_data['fix']

            train_data = train_data.drop(['fix'], axis=1)
            val_data = val_data.drop(['fix'], axis=1)
            test_data = test_data.drop(['fix'], axis=1)

            # log transformation
            train_data = np.log(train_data + 1)
            val_data = np.log(val_data + 1)
            test_data = np.log(test_data + 1)

            train_data['fix'] = train_fix
            val_data['fix'] = val_fix
            test_data['fix'] = test_fix
        else:
            # log transformation
            train_data = np.log(train_data + 1)
            val_data = np.log(val_data + 1)
            test_data = np.log(test_data + 1)

            # SMOTE
        kk = 5
        if sum(train_label) <= kk or len(train_label) - sum(train_label) <= kk:
            kk = min(sum(train_label), len(train_label) - sum(train_label)) - 1
            # print('k = ' + str(kk))

        smote = SMOTE(random_state=42, k_neighbors=kk)

        X_resampled, y_resampled = smote.fit_resample(train_data, train_label)

        train_data = X_resampled
        train_label = y_resampled

        n_features = train_data.shape[1]

        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "rb") as f:
            shap_values_loaded = pickle.load(f)

        if CLF == 'RF' or CLF == 'DT':
            shap_values_loaded = shap_values_loaded[:, :, 1]

        shap_df = pd.DataFrame(shap_values_loaded.values, columns=shap_values_loaded.feature_names)
        X_df = pd.DataFrame(shap_values_loaded.data, columns=shap_values_loaded.feature_names)


        shap_df = shap_df.reindex(columns=all_feature_names, fill_value=0)
        X_df = X_df.reindex(columns=all_feature_names, fill_value=0)

        all_shap_df_list.append(shap_df)
        all_X_df_list.append(X_df)

    shap_df_combined = pd.concat(all_shap_df_list, axis=0)
    X_df_combined = pd.concat(all_X_df_list, axis=0)

    plt.figure()
    shap.summary_plot(shap_df_combined.values, X_df_combined, show=False)

    plt.savefig(os.path.join(save_path, f"{CLF}_all_shap_plot.png"))



