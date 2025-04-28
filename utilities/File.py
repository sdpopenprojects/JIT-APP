import os
import pandas as pd


def create_dir(dirname):
    # path = os.getcwd() + '/' + dirname
    path = dirname
    folder = os.path.exists(path)

    try:
        if not folder:
            os.makedirs(path, exist_ok=True)
    except OSError as err:
        print(err)

    return path + "/"


def save_results(save_path, score):
    # with open(fres + project_name + '-' + model_name + '.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(score)

    # pandas
    tempRes = pd.DataFrame(score).T
    tempRes.to_csv(save_path + '.csv', index=False, header=False, mode='a')