import itertools

import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC, LinearSVR, NuSVR, NuSVC, SVC, SVR

def auc_value(y, pred, mode="train"):

    _dict = {
        "y_" + mode: y,
        "pred_" + mode: pred
    }
    df_pred = pd.DataFrame(_dict).sort_values(by="y_" + mode, ascending=False)

    positive_percent = y.sum()/len(y)
    df_pred["label_" + mode] = [1 if x >= np.percentile(pred, 100 - 100 * positive_percent / len(df_pred)) else 0 for x in df_pred["y_" + mode]]
    _auc = round(roc_auc_score(df_pred["y_" + mode].values, pred), 4)

    return _auc


def get_data(args):

    feats = args[0]["feats"]
    label = args[0]["label"]
    y_name = args[0]["y"]

    data_train = args[0]["data_train_valid"]

    X_train = data_train[feats].values
    X_train_add_const = sm.add_constant(X_train)
    y_train = data_train[y_name].values
    label_train = data_train[label].values

    X_test = args[0]["data_test"][feats].values
    X_test_add_const = sm.add_constant(X_test)
    y_test = args[0]["data_test"].loc[:, y_name].values
    label_test = args[0]["data_test"].loc[:, label].values

    return y_train, label_train, X_train, X_train_add_const, y_test, label_test, X_test, X_test_add_const, feats, label, args[0]["data_test"].asset


def PCA_plot(est):

    # 可视化
    plt.plot(est.explained_variance_ratio_, 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('PVE')

    plt.plot(est.explained_variance_ratio_.cumsum(), 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
    plt.title('Cumulative PVE')


def pca_test(data, feats, label):

    df_corr = pd.DataFrame(data.loc[:, feats + [label]]).corr().sort_values(by=label)
    df_corr.to_csv("df_corr.csv")
    pd.DataFrame(data.loc[:,["overnight_mom_20_0","intraday_mom_10_6"]]).corr()

    df = data.loc[:,["overnight_mom_20_0","intraday_mom_10_6"]]
    plt.scatter(df["overnight_mom_20_0"], df["intraday_mom_10_6"])
    plt.show()

    df = data.loc[:, ["dm_overnight_mom_10", "overnight_mom_10"]]
    df.loc[:, ["dm_overnight_mom_10", "overnight_mom_10"]].corr()
    plt.scatter(df["dm_overnight_mom_10"], df["overnight_mom_10"])
    plt.show()

    df = data.loc[:, ["overnight_mom_10", "intraday_mom_10"]]
    df.loc[:, ["overnight_mom_10", "intraday_mom_10"]].corr()
    plt.scatter(df["overnight_mom_10"], df["intraday_mom_10"])
    plt.show()

    df = pd.DataFrame(data[:, [124, 129]])
    plt.scatter(df[0], df[1])
    plt.show()


    df_sttdz = (df - df.mean())/df.std()

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(df_sttdz)
    pca_train_scores = pca.transform(df_sttdz)

    pca.explained_variance_ratio_


def value_df_construct(param_dict):

    y_test = param_dict["y_test"]
    y_train = param_dict["y_train"]
    pred_test = param_dict["pred_test"]
    pred_train = param_dict["pred_train"]
    date_asset = param_dict["date_asset"]

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    auc_train = round(roc_auc_score(y_train, pred_train), 4)
    auc_test = round(roc_auc_score(y_test, pred_test), 4)

    df_pred = pd.DataFrame({
        "date": date_asset.index,
        "asset": date_asset.values,
        "pred": pred_test,
        "label": y_test,
    }).sort_values(by="pred", ascending=False)
    df_pred["auc_trte_r2_trte"] = str([auc_train, auc_test, round(r2_train, 6), round(r2_test, 6)])
    df_pred.sort_values(by="pred", ascending=False, inplace=True)

    return df_pred


def get_test_data(_data, _model, feats, label):

    X_test_today = _data[feats].values
    y_test_today = _data[label].values
    y_pred_today = _model.predict(X_test_today)
    # bar_value = np.percentile(y_pred_today, 100 - 100 * y_test_today.sum() / len(y_test_today))
    # pred_label_today = [1 if x >= bar_value else 0 for x in y_pred_today]

    data_dict = {
        "date": _data.date,
        "asset": _data.asset,
        "y_test_today": y_test_today,
        "y_pred_today": y_pred_today,
        "pred_label_today": y_pred_today
    }

    return pd.DataFrame(data_dict)


def sum_value_1(models_dict, X_train, y_train, label_train, X_test, y_test, label_test, feats, args):

    evaluat_lst = []
    for _model in models_dict.values():

        pred_train = _model.predict(X_train)
        pred_test = _model.predict(X_test)

        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)

        auc_train = round(roc_auc_score(label_train, pred_train), 4)
        auc_test = round(roc_auc_score(label_test, pred_test), 4)

        evaluat_lst += [r2_train, r2_test, auc_train, auc_test]
    return evaluat_lst



def svm_value(_model, X_train, y_train, X_test, y_test, feats, label, args):


    _model.fit(X_train, y_train)
    pred_train = _model.predict(X_train)
    pred_test = _model.predict(X_test)
    label_train = args[0]["data_train_valid"][label].values
    label_test = args[0]["data_test"][label].values

    df_test = args[0]["data_test"].reset_index(drop=False).groupby("date").apply(lambda x:
                                                                                 get_test_data(x, _model, feats,
                                                                                               label)).sort_values(
        by="pred_label_today", ascending=False)

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    auc_train = round(roc_auc_score(label_train, pred_train), 4)
    auc_test = round(roc_auc_score(label_test, pred_test), 4)
    auc_test


def Grid_4_SVR(X_train, y_train, X_test, y_test, feats, label, args):

    from sklearn.model_selection import GridSearchCV

    param = {
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                'C': [0, 0.00001, 0.0001, 0.001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10],
                'degree': [1, 3, 5, 8, 10],
                'coef0': [0.0001, 0.001, 0.01, 10, 0.5],
                'gamma': (0, 1, 2),
                "epsilon": (0, 0.01, 0.03)
            },

    modelsvr = SVR()

    grids = GridSearchCV(modelsvr, param, cv=5, n_jobs=-1, verbose=2)

    grids.fit(X_train, y_train)

    grids.best_params_

    _model = SVR(
        kernel=grids.best_params_["kernel"],
        C=grids.best_params_["C"],
        degree=grids.best_params_["degree"],
        coef0=grids.best_params_["coef0"],
        gamma=grids.best_params_["gamma"],
    )
    _model.fit(X_train, y_train)

    svm_value(_model, X_train, y_train, X_test, y_test, feats, label, args)

    # _model = SVR(
    #     kernel="poly",
    #     C=5,
    #     degree=3,
    #     coef0=0.01,
    #     gamma=1,
    #     epsilon=0.01
    # )
    # _model.fit(X_train, y_train)
    # svm_value(_model, X_train, y_train, X_test, y_test, feats, label, args)



def SVM_pred(X_train, y_train, label_train, X_test, y_test, label_test, feats, args):


    C_lst = [0.000001, 0.00001, 0.001, 0.01, 0.1]
    gamma_lst = [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5]
    evaluat_tt_lst = []
    for param_i in itertools.product(C_lst, gamma_lst):
        model_R = SVR(kernel='rbf', C=param_i[0], gamma=param_i[1], epsilon=0.01)
        model_NuR = NuSVR(kernel='rbf', C=param_i[0], gamma=param_i[1])
        model_C = SVC(kernel='rbf', C=param_i[0], gamma=param_i[1])
        # model_NuC = NuSVC(kernel='rbf', C=param_i[0], gamma=param_i[1])
        model_R.fit(X_train, y_train)
        model_NuR.fit(X_train, y_train)
        model_C.fit(X_train, label_train)
        # model_NuC.fit(X_train, label_train)

        model_dict = {}
        model_dict["model_R"] = model_R
        model_dict["model_NuR"] = model_NuR
        model_dict["model_C"] = model_C
        # model_dict["model_NuC"] = model_NuC
        evaluat_lst = sum_value_1(model_dict, X_train, y_train, label_train, X_test, y_test, label_test, feats, args)
        evaluat_tt_lst.append([param_i] + evaluat_lst)

    df_params = pd.DataFrame(evaluat_tt_lst, columns=["C_gamma", "SVR_r2_train", "SVR_r2_test", "SVR_auc_train", "SVR_auc_test", "NuSVR_r2_train", "NuSVR_r2_test", "NuSVR_auc_train", "NuSVR_auc_test", "SVC_r2_train", "SVC_r2_test", "SVC_auc_train", "SVC_auc_test"])
    df_params.to_csv(index=False)


def SVC_pred(X_train, y_train, X_test, y_test, feats, label, args):

    _model = SVC(
        C=2.5,
        kernel='rbf',
        degree=3,
        gamma='auto',
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovo',
        random_state=None)

    svm_value(_model, X_train, y_train, X_test, y_test, feats, label, args)


def LSVC_pred(X_train, y_train, X_test, y_test, feats, label, args):

    LSVC = LinearSVC(
        penalty='l2',
        loss='squared_hinge',
        dual=True,
        tol=0.0001,
        C=5.0,
        multi_class='ovr',
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000)

    LSVC.fit(X_train, y_train)
    pred_train = LSVC.predict(X_train)
    pred_test = LSVC.predict(X_test)
    _model = LSVC

    df_test = args[0]["data_test"].reset_index(drop=False).groupby("date").apply(lambda x:
        get_test_data(x, _model, feats, label)).sort_values(by="pred_label_today", ascending=False)

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    auc_train = round(roc_auc_score(y_train, pred_train), 4)
    auc_test = round(roc_auc_score(y_test, pred_test), 4)
    auc_test

