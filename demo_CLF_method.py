import math
import os
import warnings
import numpy as np
import pandas as pd
import optuna
import json

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utilities.AutoSpearman import AutoSpearman
from utilities.File import create_dir, save_results
from utilities.performanceMeasure import get_measure
from utilities.rankMeasure import rank_measure
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def rf_objective(trial, train_data, train_label, val_data, val_label):
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 1, 20)

    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC
def lr_objective(trial, train_data, train_label, val_data, val_label):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    tol = trial.suggest_float('tol', 1e-3, 1e3)
    C = trial.suggest_float('C', 1e-4, 1)

    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

    model = LogisticRegression(penalty=penalty, tol=tol, C=C ,solver=solver)

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC

def knn_objective(trial, train_data, train_label, val_data, val_label,n_features):
    n_neighbors = trial.suggest_int('n_neighbors', 1, math.sqrt(n_features))
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p',1,5)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC

def nb_objective(trial, train_data, train_label, val_data, val_label):
    NBType = trial.suggest_categorical('NBType', ['gaussian', 'multinomial','bernoulli'])

    if NBType == "gaussian":
        model = GaussianNB()

    elif NBType == "multinomial":
        model = MultinomialNB()

    else:
        model = BernoulliNB()

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC

def dt_objective(trial, train_data, train_label, val_data, val_label):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 1, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    model = DecisionTreeClassifier(criterion = criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC

def gbm_objective(trial, train_data, train_label, val_data, val_label,n_features):
    n_estimators = trial.suggest_int('n_estimators', 30, 500)
    max_depth = trial.suggest_int('max_depth', 1, 15)
    max_features = trial.suggest_int('max_features', 1, math.sqrt(n_features))

    model = GradientBoostingClassifier(n_estimators =n_estimators, max_depth=max_depth, max_features=max_features)

    model.fit(train_data, train_label)
    predict_y = model.predict(val_data)
    _, _, _, _, AUC, _, _, _, MCC = get_measure(val_label, predict_y)

    return AUC

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'./result_method/GBM/'

    CLF = 'GBM' # 'LR', 'RF', 'NB','KNN','DT','GBM'

    project_names = sorted(os.listdir('./data/method/'))
    path = os.path.abspath('./data/method/')
    pro_num = len(project_names)

    column_name = ['commit_date', 'bug', 'NS', 'ND', 'NF', 'Entropy', 'Fix', 'NDEV', 'AGE', 'NUC', 'EXP', 'REXP',
                   'SEXP']
    commit_date = column_name[0]
    # gap = 2
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

        if CLF == 'RF':
            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: rf_objective(trial, train_data, train_label, val_data, val_label), n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)

            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            n_estimators = best_params['n_estimators']
            criterion = best_params['criterion']
            max_depth = best_params['max_depth']
        elif CLF == 'LR':
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            val_data = scaler.transform(val_data)
            test_data = scaler.transform(test_data)

            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: lr_objective(trial, train_data, train_label, val_data, val_label), n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            penalty = best_params['penalty']
            tol = best_params['tol']
            C = best_params['C']

        elif CLF == 'KNN':
            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: knn_objective(trial, train_data, train_label, val_data, val_label, n_features),
                           n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            n_neighbors = best_params['n_neighbors']
            weights = best_params['weights']
            p = best_params['p']

        elif CLF == 'NB':
            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: nb_objective(trial, train_data, train_label, val_data, val_label), n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            NBType = best_params['NBType']

        elif CLF == 'DT':
            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: dt_objective(trial, train_data, train_label, val_data, val_label), n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            criterion = best_params['criterion']
            max_depth = best_params['max_depth']
            min_samples_leaf = best_params['min_samples_leaf']

        elif CLF == 'GBM':
            study = optuna.create_study(direction='maximize')

            study.optimize(lambda trial: gbm_objective(trial, train_data, train_label, val_data, val_label,n_features),
                           n_trials=30,
                           n_jobs=1)

            best_params = study.best_params
            print(f"Best params: {best_params}")

            best_params_path = os.path.join(save_path, f'{project_name}_best_params.json')
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)

            n_estimators = best_params['n_estimators']
            max_depth = best_params['max_depth']
            max_features = best_params['max_features']

        for i in range(Rep):
            if CLF == 'RF':
                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
            elif CLF == 'LR':
                solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
                model = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver)
            elif CLF == 'KNN':
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
            elif CLF == 'NB':
                if NBType == 'gaussian':
                    model = GaussianNB()
                elif NBType == 'multinomial':
                    model = MultinomialNB()
                elif NBType == 'bernoulli':
                    model = BernoulliNB()
            elif CLF == 'DT':
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,random_state=Rep+1)

            elif CLF == 'GBM':
                model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                   max_features=max_features)

            model.fit(train_data, train_label)

            predict_y = model.predict(test_data)

            precision, recall, pf, f_measure, AUC, g_measure, g_mean, bal, MCC = get_measure(test_label, predict_y)

            Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rank_measure(predict_y, LOC, test_label)

            measure = [precision, recall, pf, f_measure, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall,
                       Eprecision, Efmeasure, PMI, IFA]

            save_results(save_path + project_name, measure)

    print('done!')
