import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import json

from sklearn.tree import DecisionTreeClassifier
from utilities.AutoSpearman import AutoSpearman
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'./result_method/KNN/'

    CLF = 'KNN' # 'LR', 'RF', 'NB','KNN','DT','GBM'

    project_names = sorted(os.listdir('./data/method/'))
    path = os.path.abspath('./data/method/')
    pro_num = len(project_names)

    column_name = ['commit_date', 'bug', 'NS', 'ND', 'NF', 'Entropy', 'Fix', 'NDEV', 'AGE', 'NUC', 'EXP', 'REXP',
                   'SEXP']
    commit_date = column_name[0]
    Rep = 50
    churn_name = ['commit_date', 'LA', 'LD', 'LT', 'bug']

    for i in range(0, pro_num):
        project_name = project_names[i]
        print('doing '+ project_name)
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)

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

        best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')

        # 读取 JSON 文件
        if os.path.exists(best_params_path):  # 确保文件存在
            with open(best_params_path, "r") as f:
                best_params = json.load(f)["best_params"]  # 读取并提取参数

            print("Loaded best parameters:", best_params)
        else:
            print(f"File {best_params_path} does not exist!")

        if CLF == 'RF':
            n_estimators = best_params['n_estimators']
            criterion = best_params['criterion']
            max_depth = best_params['max_depth']

            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

        elif CLF == 'LR':
            penalty = best_params['penalty']
            tol = best_params['tol']
            C = best_params['C']

            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver)

        elif CLF == 'KNN':
            n_neighbors = best_params['n_neighbors']
            weights = best_params['weights']
            p = best_params['p']

            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        elif CLF == 'NB':
            NBType = best_params['NBType']
            if NBType == 'gaussian':
                model = GaussianNB()
            elif NBType == 'multinomial':
                model = MultinomialNB()
            elif NBType == 'bernoulli':
                model = BernoulliNB()

        elif CLF == 'DT':
            criterion = best_params['criterion']
            max_depth = best_params['max_depth']
            min_samples_leaf = best_params['min_samples_leaf']

            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        elif CLF == 'GBM':
            n_estimators = best_params['n_estimators']
            max_depth = best_params['max_depth']
            max_features = best_params['max_features']

            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               max_features=max_features)
        model.fit(train_data, train_label)

        predict_y = model.predict(test_data)
            # # for explanation
        if CLF in ['KNN', 'NB', 'SVM', 'Bagging', 'AdaBoost', 'MLP']:
                # explainer = shap.Explainer(model.predict, train_data, algorithm='permutation')
            explainer = shap.KernelExplainer(model, train_data)
            # explainer = shap.PermutationExplainer(model.predict, train_data)
                # explainer = shap.SamplingExplainer(model.predict, train_data, seed=fold)
            shap_values = explainer(test_data)
        elif CLF in ['DT', 'RF', 'GBM']:
            explainer = shap.TreeExplainer(model, train_data)
            shap_values = explainer(test_data, check_additivity=False)
        elif CLF in ['LR']:
            explainer = shap.LinearExplainer(model, train_data)
            shap_values = explainer(test_data)

        shap_save_path = os.path.join(save_path, f"{project_name}_shap_values.pkl")

        with open(shap_save_path, "wb") as f:
            pickle.dump(shap_values, f)

        print(f"SHAP values saved to {shap_save_path}")
