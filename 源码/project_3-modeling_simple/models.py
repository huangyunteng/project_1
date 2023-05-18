import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import sklearn
from utils_4_models import *
from sklearn.linear_model import Lasso
import sklearn.linear_model
import os

def training(data, feats, date_lst, date_i, train_n, test_n, days_pred, model_type, direc, path_data_src):

    train_valid_days = date_lst[date_i - train_n - test_n:date_i - test_n - days_pred]
    test_days = date_lst[date_i - test_n:date_i]

    data_test = data.loc[test_days, :]
    data_train_valid = data.loc[train_valid_days, :]

    param_dict = {
        "data_train_valid": data_train_valid,
        "data_test": data_test,
        "feats": feats,
        "label": model_type["label"],
        "y": model_type["y"],
        "date_i": date_i,
        "train_n": train_n,
        "days_pred": days_pred,
        "model_type": model_type,
        "direc": direc,
        "path_data_src": path_data_src
    }

    method_name = list(model_type.values())[0]

    func = getattr(models, list(model_type.keys())[0])
    func(method_name, param_dict)


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


class models:

    def __int__(self):
        pass

    @classmethod
    def linear(cls, *args, **kwargs):

        y_train, X_train, X_train_add_const, y_test, X_test, X_test_add_const, feats, label = get_data(args)

        est1 = sm.OLS(y_train, X_train, missing="drop")  # 因为rep78有5个缺失，故少了5个观测值
        est2 = est1.fit()
        pred_train = est2.predict(X_train_add_const)
        pred_test = est2.predict(X_test_add_const)

        auc_train = auc_value(y_train, pred_train)
        auc_test = auc_value(y_test, pred_test, mode="test")

        return auc_train, auc_test


    @classmethod
    def methods_type_1(cls, method="OLS", *args, **kwargs):

        y_train, label_train, X_train, X_train_add_const, y_test, label_test, X_test, X_test_add_const, feats, label, date_asset = get_data(args)

        func = getattr(sm, method)
        est1 = func(y_train, X_train_add_const, missing="drop")  #
        est2 = est1.fit()

        pred_train = est2.predict(X_train_add_const)
        pred_test = est2.predict(X_test_add_const)

        bar = np.percentile(pred_test, 100 - 100 * sum(label_test) / len(label_test))
        pred_label = pred_test.copy()
        pred_label[pred_label > bar] = 1
        pred_label[pred_label <= bar] = 0

        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)

        auc_train = auc_value(label_train, pred_train)
        auc_test = auc_value(label_test, pred_label, mode="test")

        df_pred = pd.DataFrame({
            "date": date_asset.index,
            "asset": date_asset.values,
            "pred_test": pred_test,
            "pred_label": pred_label,
            "label": y_test,
        }).sort_values(by="pred_test",ascending=False)

        seri_coef = pd.Series(est2.params[1:], index=feats)
        seri_coef_keep = seri_coef[seri_coef != 0].sort_values(ascending=False)
        try:
            df_pred["auc_trte_r2_trte"] = str([auc_train, auc_test, round(est2.rsquared_adj, 6), round(r2_test, 6)])
        except:
            a = 0
        df_pred["_coef_keep"] = str(round(seri_coef_keep, 6).sort_values(ascending=False).to_dict())
        df_pred.sort_values(by="pred_test", ascending=False, inplace=True)

        return auc_train, auc_test


    @classmethod
    def methods_type_2(cls, method="PCA", *args, **kwargs):

        y_train, label_train, X_train, X_train_add_const, y_test, label_test, X_test, X_test_add_const, feats, label, date_asset = get_data(args)

        # from utils_4_models import pca_test
        # pca_test(args[0]["data_train_valid"], feats, label)

        func = getattr(sklearn.decomposition, method)
        est = func().fit(X_train)
        pca_train_scores = est.transform(X_train)
        pca_test_scores = est.transform(X_test)


        import matplotlib.pyplot as plt
        columns = ['P' + str(i) for i in range(len(feats))]
        plt.bar(list(np.arange(len(columns))), est.explained_variance_ratio_)
        plt.title("components percents")
        plt.show()
        plt.bar(list(np.arange(len(columns))), np.cumsum(est.explained_variance_ratio_))
        plt.title("cumsum components percents")
        plt.show()

        columns = ['PC' + str(i) for i in range(len(feats))]
        pca_loadings = pd.DataFrame(est.components_, columns=feats, index=columns)
        pca_train_scores = pd.DataFrame(pca_train_scores, columns=columns)
        pca_test_scores = pd.DataFrame(pca_test_scores, columns=columns)
        pca_train_scores = pca_train_scores.iloc[:, :100]
        pca_test_scores = pca_test_scores.iloc[:, :100]
        pca_train_scores_add_const = sm.add_constant(pca_train_scores)
        pca_test_scores_add_const = sm.add_constant(pca_test_scores)

        func = getattr(sm, "OLS")
        est1 = func(y_train, pca_train_scores_add_const, missing="drop")  # 因为rep78有5个缺失，故少了5个观测值
        est2 = est1.fit()
        pred_train = est2.predict(pca_train_scores_add_const)
        pred_test = est2.predict(pca_test_scores_add_const)

        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)

        auc_train = auc_value(label_train, pred_train)
        auc_test = auc_value(label_test, pred_test, mode="test")
        auc_test

    @classmethod
    def methods_type_3(cls, method="Lasso", *args, **kwargs):

        y_train, label_train, X_train, X_train_add_const, y_test, label_test, X_test, X_test_add_const, feats, label, date_asset = get_data(args)

        # http://www.manongjc.com/detail/31-ssilercoxbegxlh.html
        for _alpha in [0.0001, 0.0003, 0.0005, 0.001]:
            func = getattr(sklearn.linear_model, method)
            est2 = func(alpha=_alpha).fit(X_train,y_train)

            seri_coef = pd.Series(est2.coef_, index=feats)
            seri_coef_keep = seri_coef[seri_coef != 0].sort_values(ascending=False)

            est2.predict(X_test)
            pred_train = est2.predict(X_train)
            pred_test = est2.predict(X_test)

            pct = (label_test == 1).astype(int).sum() / y_test.shape[0]
            label_pred_test = pred_test.copy()
            label_pred_test[pred_test > np.percentile(pred_test, 100 - 100*pct)] = 1
            label_pred_test[pred_test <= np.percentile(pred_test, 100 - 100*pct)] = 0

            r2_train = r2_score(y_train, pred_train)
            r2_test = r2_score(y_test, pred_test)
            auc_train = auc_value(label_train, pred_train)
            auc_test = auc_value(label_test, label_pred_test, mode="test")

            df_pred = pd.DataFrame({
                "date": date_asset.index,
                "asset": date_asset.values,
                "pred_test": pred_test,
                "label_pred_test": label_pred_test,
                "y_test": y_test,
                "label_test": label_test
            }).sort_values(by="pred_test", ascending=False)
            df_pred["auc_trte_r2_trte"] = str([auc_train, auc_test, round(r2_train,6), round(r2_test,6)])
            df_pred["_coef_keep"] = str(round(seri_coef[seri_coef != 0], 6).sort_values(ascending=False).to_dict())
            df_pred.sort_values(by="pred_test", ascending=False, inplace=True)
            #
            # train_n = args[0]["train_n"]
            # days_pred = args[0]["days_pred"]
            # model_type = args[0]["model_type"]
            # direc = args[0]["direc"]
            # path_data_src = args[0]["path_data_src"]
            # date_i = args[0]["date_i"]
            #
            # path_save = path_data_src + "fold_4_df_pool\\"
            # if not os.path.exists(path_save):
            #     os.mkdir(path_save)
            #
            # file_name = "{}_{}_{}_{}_{}".format(train_n, days_pred, model_type[list(model_type.keys())[0]], direc,
            #                                     date_i)
            # df_pred.reset_index(drop=True).to_feather(path_save + file_name + ".feather")
            # df_pred.reset_index(drop=True).to_csv(path_save + file_name + ".csv", index=False)
        a = 0


    @classmethod
    def methods_type_4(cls, method="svr", *args, **kwargs):

        """
        :param method: LinearSVC,LinearSVR,NuSVC,NuSVR,OneClassSVM,SVC,SVR
        :param args:
        :param kwargs:
        :return:
        """

        y_train, label_train, X_train, X_train_add_const, y_test, label_test, X_test, X_test_add_const, feats, label, date_asset = get_data(args)

        # from sklearn.svm import LinearSVC, LinearSVR, NuSVR, NuSVC, SVC, SVR

        SVM_pred(X_train, y_train, label_train, X_test, y_test, label_test, feats, args)

        # Grid_4_SVR(X_train, y_train, X_test, y_test, feats, label, args)





        # LSVC_pred(X_train, y_train, X_test, y_test, feats, label, args)
        # SVC_pred(X_train, y_train, X_test, y_test, feats, label, args)

        # svr_rbf = SVR(kernel='rbf', C=0.1, gamma=0.1, epsilon=0.01)
        # svr_rbf.fit(X_train, y_train)
        #
        # svr_lin = SVR(kernel='linear', C=1, gamma=0.1, epsilon=0.01)
        # svr_lin.fit(X_train, y_train)
        #
        # svr_poly = SVR(kernel='poly', C=1, gamma=0.1, epsilon=0.01, degree=3)
        # svr_poly.fit(X_train, y_train)
        #
        # svc_lin = SVC(kernel='linear', C=3)
        # svc_lin.fit(X_train, y_train)

        # model = svc_lin
        # pred_train = model.predict(X_train)
        # df_test = args[0]["data_test"].reset_index(drop=False).groupby("date").apply(lambda x:
        #                 get_test_data(x, model, feats, label)).sort_values(by="pred_label_today", ascending=False)
        # param_dict = {
        #     "y_test": y_test,
        #     "y_train": y_train,
        #     "pred_test": df_test.pred_label_today.values,
        #     "pred_train": pred_train,
        #     "date_asset": date_asset,
        #     "args": args,
        #     "model": model
        # }
        # df_pred = value_df_construct(param_dict)
        # df_pred