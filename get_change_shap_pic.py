import os
import pickle
import warnings
import pandas as pd
import shap
import matplotlib.pyplot as plt

def get_all_features_name(save_path,pro_num):
    all_feature_names = set()
    for i in range(pro_num):

        project_name = project_names[i]
        project_name = project_name[:-4]

        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "rb") as f:
            shap_values_loaded = pickle.load(f)
        # print(type(shap_values_loaded))
        if CLF == 'RF' or CLF == 'DT':
            shap_values_loaded = shap_values_loaded[:, :, 1]
        features_name = shap_values_loaded.feature_names
        features_name_upper = [s.upper() for s in features_name]
        all_feature_names.update(features_name_upper)
    all_feature_names = sorted(list(all_feature_names))
    return all_feature_names

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'./result_change/LR/'

    CLF = 'LR' # 'NB','DT','GBM','KNN','LR','RF'
    # model_name = 'LR_method'

    project_names = sorted(os.listdir('./data/change/'))
    path = os.path.abspath('./data/change/')
    pro_num = len(project_names)

    column_name = ['commit_date', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp',
                   'rexp', 'sexp', 'buggy_B2']
    commit_date = column_name[0]
    # gap = 2

    churn_name = ['commit_date', 'LA', 'LD', 'LT', 'bug']
    # all_feature_names = get_all_features_name_1(pro_num)
    all_feature_names = get_all_features_name(save_path, pro_num)
    all_shap_df_list = []
    all_X_df_list = []
    for i in range(0, pro_num):
        project_name = project_names[i]
        print("doing "+ project_name)
        project_name = project_name[:-4]

        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "rb") as f:
            shap_values_loaded = pickle.load(f)

        if CLF == 'RF' or CLF == 'DT':
            shap_values_loaded = shap_values_loaded[:, :, 1]

        features_name = shap_values_loaded.feature_names
        features_name_upper = [s.upper() for s in features_name]

        shap_df = pd.DataFrame(shap_values_loaded.values, columns=features_name_upper)
        X_df = pd.DataFrame(shap_values_loaded.data, columns=features_name_upper)

        shap_df = shap_df.reindex(columns=all_feature_names, fill_value=0)
        X_df = X_df.reindex(columns=all_feature_names, fill_value=0)

        all_shap_df_list.append(shap_df)
        all_X_df_list.append(X_df)

    shap_df_combined = pd.concat(all_shap_df_list, axis=0)
    X_df_combined = pd.concat(all_X_df_list, axis=0)

    plt.figure()
    shap.summary_plot(shap_df_combined.values, X_df_combined, show=False)

    # plt.savefig(os.path.join(save_path, f"{CLF}_all_shap_plot.pdf"),format='pdf')
    plt.savefig(f"1-change-{CLF}.svg", format='svg', bbox_inches='tight')
