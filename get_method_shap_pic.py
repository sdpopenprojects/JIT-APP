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
        # print("doing " + project_name)
        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "rb") as f:
            shap_values_loaded = pickle.load(f)

        # if CLF == 'RF' or CLF == 'DT':
        #     shap_values_loaded = shap_values_loaded[:, :, 1]
        all_feature_names.update(shap_values_loaded.feature_names)
    all_feature_names = sorted(list(all_feature_names))
    return all_feature_names


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'./result_method/RF/'

    CLF = 'RF'  # 'LR', 'RF', 'NB','KNN','DT','GBM'
    # model_name = 'LR_method'

    project_names = sorted(os.listdir('./data/method/'))
    path = os.path.abspath('./data/method/')
    pro_num = len(project_names)

    column_name = ['commit_date', 'bug', 'NS', 'ND', 'NF', 'Entropy', 'Fix', 'NDEV', 'AGE', 'NUC', 'EXP', 'REXP',
                   'SEXP']
    commit_date = column_name[0]
    # gap = 2
    churn_name = ['commit_date', 'LA', 'LD', 'LT', 'bug']

    all_feature_names = get_all_features_name(save_path, pro_num)
    all_shap_df_list = []
    all_X_df_list = []

    for i in range(0, pro_num):
        project_name = project_names[i]
        print('doing ' + project_name)
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        # 按提交时间排序
        data = data.sort_values(by=['commit_date'])

        project_name = project_name[:-4]

        data['hasJavaDoc'][data['hasJavaDoc'] == False] = 0
        data['hasJavaDoc'][data['hasJavaDoc'] == True] = 1

        # All metrics
        ori_data = data.drop(column_name, axis=1)
        data['effort'] = data['LA'] + data['LD']

        # feature selection
        sel_data = AutoSpearman(ori_data)

        new_data = pd.concat([data['commit_date'], sel_data, data[['effort', 'bug']]], axis=1)

        train_data, val_data, test_data = np.split(new_data, [int(0.6 * len(new_data)), int(0.8 * len(new_data))])

        train_label = train_data['bug']
        val_label = val_data['bug']
        test_label = test_data['bug']

        train_data = train_data.drop(['commit_date', 'bug'], axis=1)
        val_data = val_data.drop(['commit_date', 'bug'], axis=1)
        test_data = test_data.drop(['commit_date', 'bug'], axis=1)

        LOC = test_data['effort']

        train_data = train_data.astype(float)
        val_data = val_data.astype(float)
        test_data = test_data.astype(float)

        # keep fix unchanged
        if 'Fix' in train_data.columns:
            train_fix = train_data['Fix']
            val_fix = val_data['Fix']
            test_fix = test_data['Fix']

            train_data = train_data.drop(['Fix'], axis=1)
            val_data = val_data.drop(['Fix'], axis=1)
            test_data = test_data.drop(['Fix'], axis=1)

            # log transformation
            train_data = np.log(train_data + 1)
            val_data = np.log(val_data + 1)
            test_data = np.log(test_data + 1)

            train_data['Fix'] = train_fix
            val_data['Fix'] = val_fix
            test_data['Fix'] = test_fix
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


