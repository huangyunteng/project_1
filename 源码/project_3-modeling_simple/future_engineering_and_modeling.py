
## -------------- 建模规划 --------------------

# 1.单个特征构建组合、评价组合
# 2.单个特征直接建模预测收益/排序，并构建组合、评价组合，评价环节效果
# 3.单个特征，构建特征工程，再直接建模，构建组合，评价组合，评价环节效果
# 4.单特征构建特征工程，测试特征工程每个单特征的效果
# 5.特征工程后，建模对比单特征与特征工程后效果的差异
# 6.增加特征，进行特征显式交叉，观察对比交叉后的建模效果
# 7.特征工程与特征交叉后，用集成模型通过进行切割式特征交叉来实现样本分类
# 8.DSIN神经网络建模
# 9.AlphaGo深度学习算法
# 10.ALphaZero深度强化学习算法
## -------------------------------------------

import os
import pandas as pd
import numpy as np
from datetime import datetime
# from multiprocessing import Pool, Manager, cpu_count
# import matplotlib.pyplot as plt
import utils
from utils import gl_dict,compute_forward_returns,get_clean_factor
import tears
from sklearn.model_selection import KFold,StratifiedKFold,cross_validate,cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingClassifier
from sklearn import svm,linear_model
from multiprocessing import Pool, Manager, cpu_count
import sm


def add_little(_data, ftr_i):
    _data[ftr_i] = _data[ftr_i] + [x/10000000*_data[ftr_i].values[0]  if _data[ftr_i].values[0] > 0 else x/10000000 for x in range(0,len(_data))]
    return _data

def make_factor_different(factor, ftr_i):
    factor = factor.groupby(ftr_i).apply(lambda y:add_little(y, ftr_i))
    return factor

def get_factor_price(df_ftrs1, ftr):
    df_ftr = df_ftrs1.loc[:, ["date", "asset", "factor_price"] + ftr].dropna(axis=0)

    # for ftr_i in ftr:
    #     factors = df_ftr.loc[:, ["date", "asset"] + ftr].groupby("date").apply(lambda x: make_factor_different(x, ftr_i)).set_index(["date", "asset"])

    factors = df_ftr.loc[:, ["date", "asset"] + ftr].set_index(["date", "asset"])
    prices = df_ftr.loc[:, ["date", "asset", "factor_price"]].pivot(columns="asset", values="factor_price", index="date")
    return factors, prices


def get_factor_data(factors, prices):
    quantiles = 7
    periods = (1, 3, 5, 10, 20)
    filter_zscore = 20
    cumulative_returns = True
    groupby_labels = None
    groupby = None
    bins = None
    binning_by_group = False
    max_loss = 0.35
    zero_aware = False
    forward_returns = compute_forward_returns(factors, prices, periods, filter_zscore, cumulative_returns)

    factor_data = get_clean_factor(factors, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)
    return factor_data


def get_one_label(data):

    print("date:{}".format(data.date.values[0]))
    for ftr in ["1D", "3D", "5D", "10D", "20D"]:
        tmp = np.percentile(data[ftr], 85)
        data[ftr + "_label"] = [1 if x > tmp else 0 for x in data[ftr]]
        data[ftr] = (data[ftr] - data[ftr].mean()) / data[ftr].std()
    return data

def get_clsfy_label(factor_data):
    factor_data = factor_data.groupby("date").apply(lambda x: get_one_label(x))
    return factor_data


# def make_factor_different(factor):
#     factor = factor.groupby("factor_value").apply(lambda y:add_little(y))
#     return factor


def ftr_test(df_ftrs1, ftr, _dict):

    df_ftr = df_ftrs1.loc[:, ["date", "asset", ftr, "factor_price"]].rename(columns={ftr: "factor_value"}).dropna(
        axis=0)
    factors = df_ftr.loc[:, ["date", "asset", "factor_value"]].groupby("date").apply(lambda x: make_factor_different(x,ftr)).set_index(["date", "asset"])
    # factors = df_ftr.loc[:, ["date", "asset", "factor_value"]].set_index(["date", "asset"])
    prices = df_ftr.loc[:, ["date", "asset", "factor_price"]].pivot(columns="asset", values="factor_price",
                                                                    index="date")


def get_rtn(_data):
    try:
        ptl = _data.groupby("factor_quantile").apply(lambda x: x[["1D", "3D", "5D", "10D", "20D"]].mean())
    except:
        a = 0
    return ptl


def get_nv(_data, cc="3D"):
    nv = np.cumsum(_data.sort_values(by="date")[cc])  # [["1D","3D","5D","10D","20D"]])
    nv = nv.reset_index(drop=True).reset_index(drop=False)
    return nv

import matplotlib.pyplot as plt


def stddz(_data):

    def _stddz(_data1):
        return (_data1 - _data1.mean()) / _data1.std()
    data_sttdz = _data.apply(lambda y: _stddz(y))

    return data_sttdz


def plot_factors(df_sttdz_FE_dm_tt, df_ftrs1, ftr):

    factors = df_sttdz_FE_dm_tt.loc[:, [ftr]].rename(columns={ftr: "factor_value"}).dropna(axis=0)
    factors_price = pd.merge(factors, df_ftrs1.loc[:, ["date", "asset", "factor_price"]].set_index(["date", "asset"]),
                             left_index=True, right_index=True)
    prices = factors_price.loc[:, ["factor_price"]].reset_index(drop=False).pivot(columns="asset",
                                                                                  values="factor_price", index="date")

    gl_dict["NAME_FTR"] = ftr
    gl_dict["RTN_EXIST"] = False

    path_new = "./factors_info_eng/"
    path_ftr = "./factors_info_eng/" + gl_dict["NAME_FTR"] + "/"
    path_cache = path_ftr + "DATA_CACHE/"
    path_plot = path_ftr + "PLOTS/"
    path_stat = path_new + "STAT_FUT/"
    path_plot_together = path_stat + "PLOTS_TOGATHER/"
    for _path in [path_new, path_ftr, path_cache, path_plot, path_stat, path_plot_together]:
        if not os.path.exists(_path):
            os.mkdir(_path)

    gl_dict["PATH_FTR"] = path_ftr
    gl_dict["PATH_CACHE"] = path_cache
    gl_dict["PATH_PLOT"] = path_plot
    gl_dict["PATH_PLOT_TOGATHER"] = path_plot_together

    values_num = len(factors.factor_value.drop_duplicates())
    if values_num <= 7:
        quantiles = values_num
    else:
        quantiles = 5

    periods = (1, 3, 5, 10, 20)
    filter_zscore = 20
    cc = "3D"
    cumulative_returns = True
    groupby_labels = None
    groupby = None
    bins = None
    binning_by_group = False
    max_loss = 0.35
    zero_aware = False
    forward_returns = compute_forward_returns(factors, prices, periods, filter_zscore, cumulative_returns)

    if values_num <= 7:
        factor_data = pd.merge(forward_returns, factors, left_index=True, right_index=True)
        factor_data["factor_quantile"] = factor_data.factor_value + 1
    else:
        factor_data = get_clean_factor(factors, forward_returns, groupby=groupby,
                                       groupby_labels=groupby_labels,
                                       quantiles=quantiles, bins=bins,
                                       binning_by_group=binning_by_group,
                                       max_loss=max_loss, zero_aware=zero_aware)
    factor_data.columns = ["1D", "3D", "5D", "10D", "20D", "factor_value", "factor_quantile"]
    df_ptl_rtn = factor_data.groupby("date").apply(lambda x: get_rtn(x)).reset_index(drop=False)
    df_nv = df_ptl_rtn.groupby("factor_quantile").apply(lambda x: get_nv(x, cc))
    df_nv = df_nv.reset_index(drop=False)
    df_nv = df_nv.pivot(columns="factor_quantile", index="index", values=cc)
    df_nv["dif" + str(quantiles) + "_1"] = df_nv[quantiles] - df_nv[1.0]
    df_nv["dif1_" + str(quantiles)] = df_nv[1.0] - df_nv[quantiles]
    for col in df_nv.columns:
        plt.plot(df_nv[col], label=str(col))
        plt.legend(loc='upper left', prop={'size': 9})
    plt.title(ftr)
    plt.savefig(path_stat + ftr + "_" + cc + ".png")
    plt.close()

    df_nv.to_csv(ftr + ".csv")
    df_nv
    # data = utils.get_clean_factor_and_forward_returns(factors, prices, quantiles=5, periods=(1, 3, 5, 10, 20))
    # tears.create_full_tear_sheet(data)


if __name__ == "__main__":


    # 1 数据准备
    # path = "H:\\AlphaLens_hyt_20230112\\future\\"
    path = "D:\\work\\projects\\AlphaLens_hyt_20230112\\future\\"
    df_ftrs1 = pd.read_feather(path + "df_ftrs1.feather")
    df_ftrs1["date"] = df_ftrs1["date"].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d"))
    ftrs1 = df_ftrs1.columns[:-3].tolist()

    feat_lst = ["overnight_mom_20", "overnight_mom_200", "sharpe_20", "rtn_10", "sharpe_10", "intraday_mom_10", "mk_rto_20", "rtn2md_10", "overnight_mom_10", "posi_40", "std_200"]
    # factors, prices = get_factor_price(df_ftrs1, feat_lst)
    # factors.reset_index(drop=False).to_feather("factors.feather")
    # prices.reset_index(drop=False).to_feather("prices.feather")
    #
    # factors = pd.read_feather("factors.feather").set_index(["date", "asset"])
    # prices = pd.read_feather("prices.feather").set_index(["date"])
    # for _i in range(len(feat_lst)):
    #     print(_i)
    #     ftr_i = feat_lst[_i]
    #     factor_data = get_factor_data(factors[ftr_i], prices).reset_index(drop=False)
    #     if _i == 0:
    #         factors_data = factor_data.rename(columns={"factor": ftr_i, "factor_quantile": ftr_i + "__quantile"})
    #     else:
    #         tmp = factor_data.rename(columns={"factor": ftr_i, "factor_quantile": ftr_i + "__quantile"}).loc[:, ["date", "asset", ftr_i, ftr_i + "__quantile"]]
    #         factors_data = pd.merge(factors_data, tmp, on=["date", "asset"], how="left")
    #
    # factors_data = get_clsfy_label(factors_data)
    # factors_data.to_feather("feats_data.feather")

    factors_data = pd.read_feather("feats_data.feather")

    factors_data_stddz = factors_data[["date", "asset"] + feat_lst].groupby("date").apply(lambda x: stddz(x.set_index(["date", "asset"])))
    factors_data_stddz.index = factors_data_stddz.index.droplevel(0)
    factors_data = factors_data_stddz.reset_index(drop=False)

    # from feature_engineer import FeatureEngineer
    # FE = FeatureEngineer()
    # df_method_4th = factors_data[["date", "asset"] + feat_lst].groupby("date").apply(lambda x: FE._4th_method(x.set_index(["date", "asset"]), trans_type=1))
    # df_method_4th.index = df_method_4th.index.droplevel(0)
    #
    # df_FE_dm = factors_data[["date", "asset"] + feat_lst].groupby("date").apply(lambda x: FE.get_my_dummies(x.set_index(["date", "asset"]), cuts=5)).fillna(0)
    # df_FE_dm.index = df_FE_dm.index.droplevel(0)
    #
    # df_sttdz_FE_dm_tt = pd.concat([factors_data.set_index(["date", "asset"]), df_method_4th, df_FE_dm], axis=1)
    #
    # for _var in ["1D","3D","5D","10D","20D"]:
    #     df_sttdz_FE_dm_tt[_var] = factors_data.set_index(["date","asset"])[_var]
    #
    # df_sttdz_FE_dm_tt.reset_index(drop=False).to_feather("df_sttdz_FE_dm_tt.feather")
    #
    df_sttdz_FE_dm_tt = pd.read_feather("df_sttdz_FE_dm_tt.feather").set_index(["date", "asset"])
    #
    # # 单进程
    # for ftr in df_sttdz_FE_dm_tt.columns:
    #     print("ftr:{}".format(ftr))
    #     plot_factors(df_sttdz_FE_dm_tt, df_ftrs1, ftr)
    #
    # # 进程池
    # poo = Pool(cpu_count()-1)
    # for ftr in df_sttdz_FE_dm_tt.columns:
    #     print("ftr:{}".format(ftr))
    #     try:
    #         plot_factors(df_sttdz_FE_dm_tt, df_ftrs1, ftr)
    #     except:
    #         print(" --------- sth wrong --------- ")
    #     poo.apply_async(plot_factors, (df_sttdz_FE_dm_tt, df_ftrs1, ))
    # poo.close()
    # poo.join()

    df_sttdz_FE_dm_tt = df_sttdz_FE_dm_tt.dropna().fillna(0).reset_index().set_index("date")
    ftr = df_sttdz_FE_dm_tt.columns.tolist()[1:-5]
    date_lst = df_sttdz_FE_dm_tt.index.get_level_values(0).drop_duplicates().sort_values().tolist()
    total_valid = 2
    train_n = 250
    valid_n = int(train_n / total_valid)
    test_n = 1

    days_pred = 5
    label = str(days_pred) + "D"

    for date_i in range(train_n + valid_n + days_pred + test_n, len(date_lst)-1):
        print("date:{}".format(date_lst[date_i]))
        train_valid_days = date_lst[date_i-train_n-test_n:date_i-test_n-days_pred]
        test_days = date_lst[date_i-test_n:date_i]

        data_test = df_sttdz_FE_dm_tt.loc[test_days, :]
        data_train_valid = df_sttdz_FE_dm_tt.loc[train_valid_days, :]






        kfold = KFold(n_splits=valid_n)

        model_score_dict = {}
        my_model = RandomForestRegressor(n_estimators=80)

        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
        model_score_dict["RF"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}

        my_model = svm.SVR()
        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
        model_score_dict["SVR"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}

        my_model = linear_model.LinearRegression()
        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
        model_score_dict["linear"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}
        min_impurity_split = None,
        S_kfold = StratifiedKFold(n_splits=valid_n)


        my_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                                verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                                max_samples=None)

        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
                             scoring=['neg_log_loss','roc_auc'], error_score='raise')
        model_score_dict["RF_c"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}

        # my_model = svm.SVC()
        # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
        #                      scoring=['neg_log_loss','roc_auc'], error_score='raise')
        # model_score_dict["SVR_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}

        my_model = linear_model.LogisticRegression()
        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
                             scoring=['neg_log_loss','roc_auc'], error_score='raise')
        model_score_dict["Logistic_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}

        my_model = GradientBoostingClassifier()
        tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold, scoring=['neg_log_loss', 'roc_auc'], error_score='raise')
        model_score_dict["GB_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}

        print(model_score_dict)

    # 1.1 样本构建：N天收益和排序，0-1标记，训练、验证、测试数据集划分和利用


    # 1.单个特征构建组合、评价组合
