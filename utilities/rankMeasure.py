import numpy as np


# effort-aware performance measures
def rank_measure(predict_score, effort, test_label):
    length = len(test_label)

    effort = np.array(effort)
    if 0 in effort:
        # for avoiding effort has zero
        effort = effort + 1

    predict_label = predict_score
    predict_label[predict_label >= 0.5] = 1
    predict_label[predict_label < 0.5] = 0

    # predict defect density
    pred_density = predict_score / effort
    actual_density = test_label / effort

    # combining
    data = np.zeros(shape=(len(test_label), 5))
    data[:, 0] = predict_label
    data[:, 1] = pred_density
    data[:, 2] = test_label
    data[:, 3] = actual_density
    data[:, 4] = effort

    # actual model(CBS+)
    data_mdl = sorted(data, key=lambda x: (-x[0], -x[1]))  # x[0]:predict_label  x[1]: pred_density
    data_mdl = np.array(data_mdl)
    mdl = computeArea(data_mdl, length)

    # optimal model
    data_opt = sorted(data, key=lambda x: (-x[3], x[4]))  # x[3]:actual_density  x[4]: effort
    data_opt = np.array(data_opt)
    opt = computeArea(data_opt, length)

    # worst model
    data_wst = sorted(data, key=lambda x: (x[3], -x[4]))  # x[3]: actual_density  x[4]: effort
    data_wst = np.array(data_wst)
    wst = computeArea(data_wst, length)

    if opt - wst != 0:
        Popt = 1 - (opt - mdl) / (opt - wst)
    else:
        Popt = 0.5

    cErecall, cEprecision, cEfmeasure, cPMI, cIFA = computeMeasure(data_mdl, length)

    return Popt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA


def computeMeasure(data, length):
    cumXs = np.cumsum(data[:, 4])  # effort
    cumYs = np.cumsum(data[:, 2])  # test_label
    Xs = cumXs / cumXs[length - 1]  # percent of effort

    idx = np.min(np.where(Xs >= 0.2))
    pos = idx + 1

    Erecall = cumYs[idx] / cumYs[length - 1]

    Eprecision = cumYs[idx] / pos

    if Erecall + Eprecision != 0:
        Efmeasure = 2 * Erecall * Eprecision / (Erecall + Eprecision)
    else:
        Efmeasure = 0

    PMI = pos / length

    Iidx = np.min(np.where(cumYs >= 1))
    IFA = Iidx + 1

    return Erecall, Eprecision, Efmeasure, PMI, IFA


def computeArea(data, length):
    data = np.array(data)
    cumXs = np.cumsum(data[:, 4])  # effort
    cumYs = np.cumsum(data[:, 2])  # test_label

    Xs = cumXs / cumXs[length - 1]
    Ys = cumYs / cumYs[length - 1]

    # Use the trapezoidal rule to calculate the area under the curve
    fix_subareas = [0] * len(Ys)
    fix_subareas[0] = 0.5 * Ys[0] * Xs[0]
    for i in range(1, len(Ys)):
        fix_subareas[i] = 0.5 * (Ys[i - 1] + Ys[i]) * abs(Xs[i - 1] - Xs[i])

    area = sum(fix_subareas)

    return area
