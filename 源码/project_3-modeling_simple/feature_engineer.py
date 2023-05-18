import numpy as np
import pandas as pd
# import toad
from datetime import datetime
# import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

class FeatureEngineer:

    def __init__(self):
        pass

    def get_numeric_features(self, data):
        set_drop = set([""])
        numeric_features = [x for x in list(data.keys()) if len(x & set_drop) == 0]
        return numeric_features

    def get_category_features(self, data):
        set_drop = set([""])
        category_features = [x for x in list(data.keys()) if len(x & set_drop) == 0]
        return category_features

    def _4th_method(self, data, trans_type=1):

        littl = 1 / 100000000000000

        def get_softmax_1(_distri, ptl_lst):
            tmp = [1 / abs(_distri - _a + littl) for _a in ptl_lst]
            w_ = [x / sum(tmp) for x in tmp]
            return w_

        def get_softmax_2(_distri, ptl_lst):
            tmp = [(1 / (_distri - _a + littl)) ** 2 for _a in ptl_lst]
            w_ = [x / sum(tmp) for x in tmp]
            return w_

        # 1 4th method
        def one_feature(_x):

            ptl_lst = [np.percentile(_x, p_th) for p_th in [0,10,20,30,40,50,60,70,80,90]]
            if trans_type == 0:
                w_ = list(_x.apply(lambda _z: get_softmax_1(_z, ptl_lst)))
            elif trans_type == 1:
                w_ = list(_x.apply(lambda _z: get_softmax_2(_z, ptl_lst)))
            return w_

        tmp = data.apply(lambda x:one_feature(x))
        cols = tmp.columns.tolist()
        data_4th_train = pd.concat(
            [pd.DataFrame(list(tmp[x]), index=tmp.index, columns=[x + "_" + str(_i) for _i in range(0, 10)]) for x in
             cols], axis=1)

        return data_4th_train

    # def woe(self, X_train, label_train, X_test, label_test):
    #
    #     # 2 woe
    #     X_train["label"] = label_train["label"]
    #     c = toad.transform.Combiner()
    #     c.fit(X_train, y='label', method='chi', min_samples=0.05)  # empty_separate = False
    #     c.transform(X_train, label=False)
    #     transer = toad.transform.WOETransformer()
    #     # combiner.transform() & transer.fit_transform() 转化训练数据，并去掉target列
    #     train_woe = transer.fit_transform(c.transform(X_train), label_train.values)
    #     test_woe  = transer.transform(c.transform(X_test))
    #
    #     # GBDT + LR的树模型特征输出
    #     gbdt_transer = toad.transform.GBDTTransformer()
    #     gbdt_transer.fit(X_train, 'label', n_estimators=10, max_depth=2)
    #     gbdt_vars = gbdt_transer.transform(X_test)
    #     return train_woe, test_woe, gbdt_vars

    def get_my_dummies(self, data, cuts=10):

        if data.index.values[0][0].date() == datetime.strptime("2010-11-04","%Y-%m-%d").date():
            a = 0
        if len(data.dropna()) < cuts:
            return
        else:
            def get_1_dummy(col, feat):
                try:
                    data_cut = pd.cut(feat, cuts)
                except:
                    print("sth wrong with")
                return data_cut

            df = (pd.Series(data.columns).apply(lambda x: get_1_dummy(x, data[x])))
            df = df.T
            df.columns = ["dm_" + x for x in list(data.columns)]

            # LabelEncoding编码
            for col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col]).astype(int)
            df_encode = df

            for col in df.columns:
                df = pd.get_dummies(df, columns=[col])

            return df_encode.join(df)

    # def box_onehot(self, data, cuts=10):
    #     def box_1_feat(feat):
    #         pct_lst = [np.percentile(feat, x) for x in range(1,cuts)]
    #         def get_pos(pct_lst, _z):
    #             [_z / _p for _p in pct_lst]
    #             a = 0
    #         feat.apply(lambda _y: get_pos(pct_lst,_y))    #
    #     data.apply(lambda x:x)

    # def get_iv_gini(self, data):
    #     to_drop = []
    #     res = toad.quality(data.drop(to_drop, axis=1), 'target')
    #     return res

    # def get_onehot_map(self):
    #     a = 0

    def lasso(self, X_train, label_train, X_test,label_test, alpha=0.1):
        from sklearn.linear_model import Lasso
        from sklearn.metrics import r2_score
        lasso = Lasso(alpha=alpha)
        model = lasso.fit(X_train, label_train)
        label_pred = model.predict(X_test)
        r2_score_lasso = r2_score(label_test, label_pred)
        return r2_score_lasso, model

    # def boxes(self, data_train, data_test):
    #     # initialise
    #     c = toad.transform.Combiner()
    #
    #     # 使用特征筛选后的数据进行训练：使用稳定的卡方分箱，规定每箱至少有5%数据, 空值将自动被归到最佳箱。
    #     to_drop = []
    #     c.fit(data_train.drop(to_drop, axis=1), y='target', method='chi',min_samples=0.05)  # empty_separate = False
    #
    #     # 为了演示，仅展示部分分箱
    #     print('var_d2:', c.export()['var_d2'])
    #     print('var_d5:', c.export()['var_d5'])
    #     print('var_d6:', c.export()['var_d6'])
    #
    #     # 初始化
    #     transer = toad.transform.WOETransformer()
    #
    #     # combiner.transform() & transer.fit_transform() 转化训练数据，并去掉target列
    #     train_woe = transer.fit_transform(c.transform(data_train), data_train['target'],
    #                                       exclude=to_drop + ['target'])
    #     woe_test = transer.transform(c.transform(data_test))
    #
    #     print(train_woe.head(3))
    #
    #     return woe_test

    # def stepwise(self, woe_train, woe_test):
    #     to_drop = []
    #     # 将woe转化后的数据做逐步回归
    #     final_data = toad.selection.stepwise(woe_train, target='target', estimator='lasso', direction='both',
    #                                          criterion='aic', exclude=to_drop)
    #     # 将选出的变量应用于test/OOT数据
    #     final_test = woe_test[final_data.columns]
    #
    #     print(final_test.shape)  # 逐步回归从31个变量中选出了10个
    #     # 确定建模要用的变量
    #     col = list(final_data.drop(to_drop + ['target'], axis=1).columns)
    #     toad.metrics.PSI(final_data[col], final_test[col])
    #     return final_test

    def _xgb(self, data):
        xgb_train = xgb.DMatrix(batch_xs, label=batch_ys)
        xgb_test = xgb.DMatrix(test_crtr, label=test_label)
        params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eta": 0.1,
            "max_depth": 5
        }

        # watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
        num_round = 100
        res = xgb.cv(params, xgb_train, num_round, nfold=5, metrics={"auc"}, seed=0,
                     callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


# import pandas as pd
# data_train = pd.read_csv("./data/data_train_less.csv").set_index(["date","fund_code"])
# FE = FeatureEngineer()
# tmp = FE._4th_method(data_train)
# data_4th_train