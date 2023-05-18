import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from models import training
import itertools


def get_one_label(data):
    percent_setting = 20
    for ftr in ["1D", "3D", "5D", "10D", "20D"]:
        tmp = np.percentile(data[ftr], 100 - percent_setting)
        data["label_" + ftr +"_positive"] = [1 if x > tmp else 0 for x in data[ftr]]
        tmp = np.percentile(data[ftr], percent_setting)
        data["label_" + ftr + "_negative"] = [1 if x < tmp else 0 for x in data[ftr]]
    return data


def get_clsfy_label(factor_data):
    factor_data = factor_data.groupby("date").apply(lambda x: get_one_label(x))
    return factor_data


def get_src_data_and_stddz(path_of_data_src):

    src_data_name = "df_sttdz_FE_dm_tt.feather"
    df_sttdz_FE_dm_tt = pd.read_feather(path_of_data_src + src_data_name)

    df_sttdz_FE_dm_tt["date"] = df_sttdz_FE_dm_tt.date.apply(lambda x: x.date())
    df_sttdz_FE_dm_tt = df_sttdz_FE_dm_tt.set_index(["date", "asset"]).dropna().fillna(0).reset_index().set_index("date")
    ftr = df_sttdz_FE_dm_tt.columns.tolist()[1:-5]
    date_lst = df_sttdz_FE_dm_tt.index.get_level_values(0).drop_duplicates().sort_values().tolist()

    df_sttdz_FE_dm_tt.reset_index(drop=False, inplace=True)

    def get_sttdz(_data, trans_type=0):
        _data = _data.set_index("asset").drop("date", axis=1)
        if trans_type == 0:
            return (_data - _data.mean()) / _data.std()
        elif trans_type == 1:
            return (_data - _data.min()) / (_data.max() - _data.min())

    df_sttdz_FE_dm_tt_1 = df_sttdz_FE_dm_tt.set_index(["date", "asset"])
    df_sttdz_FE_dm_tt_1[ftr] = df_sttdz_FE_dm_tt[["date", "asset"] + ftr].groupby("date").apply(lambda x: get_sttdz(x)).fillna(0)

    return date_lst, ftr, df_sttdz_FE_dm_tt_1


days_pred = 3
model_type_lst = [
        {
            "methods_type_4":"svr",
            "y": str(days_pred) + "D",
            "label": "label_" + str(days_pred) +"D"
        },
        # {
        #     "methods_type_3":"Lasso",
        #     "y": str(days_pred) + "D",
        #     "label": "label_" + str(days_pred) +"D"
        # },
        # {
        #     "methods_type_1": "Logit",
        #     "y": str(days_pred) + "D",
        #     "label": "label_" + str(days_pred) +"D"
        # },
        # {
        #     "methods_type_2": "PCA",
        #     "y": str(days_pred) + "D",
        #     "label": "label_" + str(days_pred) + "D"
        # },
        # {
        #     "methods_type_1": "OLS",
        #     "y": str(days_pred) + "D",
        #     "label": "label_" + str(days_pred) + "D"
        #  }
    ]

if __name__ == "__main__":

    path_data_src = "D:\\work\\projects\\Data\\project_3-modeling_simple\\"
    date_lst, ftr, df_sttdz_FE_dm_tt = get_src_data_and_stddz(path_data_src)
    df_sttdz_FE_dm_tt_ = get_clsfy_label(df_sttdz_FE_dm_tt.reset_index(drop=False))
    df_sttdz_FE_dm_tt_.set_index("date", inplace=True)

    valid_n = 0
    test_n = 1

    train_n_lst = [160, 20, 30, 60, 80, 90, 120, 200, 250]
    direc_lst = ["positive"] #"negative",

    pro_dict = {}
    for train_n, days_pred, model_type, direc in itertools.product(train_n_lst, [days_pred], model_type_lst, direc_lst):

        poo = Pool(cpu_count() - 3)
        for date_i in range(train_n + valid_n + days_pred + test_n, len(date_lst)-1):
            print(f"date, train_n, days_pred, model_type, direc:{date_lst[date_i], train_n, days_pred, model_type, direc}")

            if direc == "positive":
                cols_keep = [x for x in df_sttdz_FE_dm_tt_.keys() if x.split("_")[len(x.split("_"))-1] != "negative"]
                cols_name = [x[:-9] if x.split("_")[len(x.split("_"))-1] == "positive" else x for x in
                             cols_keep]

            elif direc == "negative":
                cols_keep = [x for x in df_sttdz_FE_dm_tt_.keys() if x.split("_")[len(x.split("_"))-1] != "positive"]
                cols_name = [x[:-9] if x.split("_")[len(x.split("_")) - 1] == "negative" else x for x in
                             cols_keep]
            df_sttdz_FE_dm_tt_1 = df_sttdz_FE_dm_tt_.loc[:, cols_keep]
            df_sttdz_FE_dm_tt_1.columns = cols_name

            training(df_sttdz_FE_dm_tt_1, ftr, date_lst, date_i, train_n, test_n, days_pred, model_type, direc, path_data_src,)
            # poo.apply_async(training, (df_sttdz_FE_dm_tt_1, ftr, date_lst, date_i, train_n, test_n, days_pred, model_type, direc, path_data_src,))
        # poo.close()
        # poo.join()











    # kfold = KFold(n_splits=valid_n)
    #
    # model_score_dict = {}
    # my_model = RandomForestRegressor(n_estimators=80)
    #
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
    # model_score_dict["RF"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}
    #
    # my_model = svm.SVR()
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
    # model_score_dict["SVR"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}
    #
    # my_model = linear_model.LinearRegression()
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=kfold, scoring=['r2', 'neg_mean_squared_error'], error_score='raise')
    # model_score_dict["linear"] = {"test_r2": tmp["test_r2"].mean(), "test_neg_mean_squared_error": tmp["test_neg_mean_squared_error"].mean()}
    # min_impurity_split = None,
    # S_kfold = StratifiedKFold(n_splits=valid_n)
    #
    #
    # my_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
    #                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
    #                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                                         bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
    #                                         verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
    #                                         max_samples=None)
    #
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
    #                      scoring=['neg_log_loss','roc_auc'], error_score='raise')
    # model_score_dict["RF_c"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}
    #
    # # my_model = svm.SVC()
    # # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
    # #                      scoring=['neg_log_loss','roc_auc'], error_score='raise')
    # # model_score_dict["SVR_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}
    #
    # my_model = linear_model.LogisticRegression()
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold,
    #                      scoring=['neg_log_loss','roc_auc'], error_score='raise')
    # model_score_dict["Logistic_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}
    #
    # my_model = GradientBoostingClassifier()
    # tmp = cross_validate(my_model, data_train_valid[ftr], data_train_valid[label], cv=S_kfold, scoring=['neg_log_loss', 'roc_auc'], error_score='raise')
    # model_score_dict["GB_C"] = {"neg_log_loss": tmp["test_neg_log_loss"].mean(), "roc_auc": tmp["test_roc_auc"].mean()}
    #
    # print(model_score_dict)
    #
    # # 1.1 样本构建：N天收益和排序，0-1标记，训练、验证、测试数据集划分和利用
    #
    #
    # # 1.单个特征构建组合、评价组合
