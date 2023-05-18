# 上一版本模型NFold交叉验证，证明了 特征工程 特征交叉 的有效性，但个人认为：训练中把不同交易日的样本放在一起，造成与实际不符的数据分布，可能有影响。
# 验证的时候也是很多天放在一起，它模型预测时是否有影响呢？所以，我认为应该尊重实际的数据分布，每天作为一个随机梯度样本输入训练；验证时，也是每天预测一
# 次，并每天选一次概率靠前的票预测label置为1；或一次性预测回归值，但每天去统计靠前的pre_label，然后再计算auc。否则与实际情况总是有出入，可能有些天
# 都没有预测出有持仓的情况。

import os.path

from Logistic.model import Logistic
from FM.model import FM
from DCN.model import DCNWithOutEmb
from xDeepFM.model import xDeepFMWithoutEmb
# from DeepCrossing.model import DeepCrossingH

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_data():
    src_data_path = "D:\\work\\projects\\AlphaLens_hyt_20230112\\future\\"
    data_name = "df_sttdz_FE_dm_tt.feather"
    df_sttdz_FE_dm_tt = pd.read_feather(src_data_path + data_name)
    data_name = "feats_data.feather"
    feats_data = pd.read_feather(src_data_path + data_name)
    df = pd.merge(df_sttdz_FE_dm_tt, feats_data.loc[:, ["date", "asset", "3D_label", "5D_label"]], on=["date", "asset"], how="left")
    return df


def model_value(model, X_valid, y_valid):
    # 评估
    pre = model(X_valid)
    AUC = round(roc_auc_score(y_valid, pre), 4)
    return AUC


def define_model(param_dict):

    model_type = param_dict["model_type"]
    feature_columns = param_dict["feature_columns"]

    if model_type == "logistic":
        _model = Logistic(k=8, w_reg=1e-4)

    elif model_type == "fm":
        _model = FM(k=8, w_reg=1e-4, v_reg=1e-4)

    elif model_type == "dcn":
        hidden_units = [256, 128, 64]
        _model = DCNWithOutEmb(hidden_units, 1, activation='relu', layer_num=6)
    elif model_type == "xDeepFMWithoutEmb":
        hidden_units = [256, 128, 64]
        dropout = 0.3
        cin_size = [128, 128]
        _model = xDeepFMWithoutEmb(cin_size, hidden_units, dropout=dropout)
    elif model_type == "DeepCrossing":
        k = 32
        hidden_units = [256, 256]
        res_layer_num = 4

        # _model = DeepCrossingH(feature_columns, k, hidden_units, res_layer_num)

    return _model


def train_model(data_train, X_col, y_col, model_type="logistic", epoches=3):

    optimizer = optimizers.SGD(0.01)
    param_dict = {
        "model_type":model_type,
        "feature_columns":""
    }
    _model = define_model(param_dict)
    mms = StandardScaler()
    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(epoches):
        for data_batch in data_train.groupby("date"):
            date = data_batch[0]
            X_train = mms.fit_transform(data_batch[1][X_col])
            y_train = data_batch[1][y_col].values
            with tf.GradientTape() as tape:
                y_pre = _model(X_train)
                _loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            with summary_writer.as_default():
                tf.summary.scalar("loss", _loss, step=i)
            grad = tape.gradient(_loss,_model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grad, _model.variables))
        print("date:{}, loss:{}".format(date.date(), round(_loss.numpy(), 5)))
    return _model, _loss.numpy()


def train_and_test(_data_train_valid, _data_test, X_col, y_col, model_type="logistic", epoches=6):

    model_name = model_type
    _model, _loss = train_model(_data_train_valid, X_col, y_col, model_type=model_type, epoches=epoches)
    auc_train = model_value(_model, _data_train_valid[X_col].values, _data_train_valid[y_col].values)
    auc_test = model_value(_model, _data_test[X_col].values, _data_test[y_col].values)
    print("auc_train:{}, auc_test:{}".format(auc_train, auc_test))


# from sklearn import preprocessing
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from DSIN.train_dsin import train_dsin
# from tqdm import tqdm
# from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# def get_feature_param(data_tt, date, ID_OFFSET, X_col):
#
#     SESS_COUNT = 5
#     DSIN_SESS_MAX_LEN = 10
#
#     sparse_features = [x for x in X_col if x.split("_")[0] == "dm"]
#     dense_features = [x for x in X_col if x.split("_")[0] != "dm"]
#     for feat in tqdm(sparse_features):
#         lbe = LabelEncoder()  # or Hash
#         data_tt[feat] = lbe.fit_transform(data_tt[feat])
#     mms = StandardScaler()
#     data_tt[dense_features] = mms.fit_transform(data_tt[dense_features])
#
#     sparse_feature_list = [SparseFeat(feat, vocabulary_size=data_tt[feat].max(
#     ) + ID_OFFSET) for feat in sparse_features + ['asset_num']]
#     dense_feature_list = [DenseFeat(feat, dimension=1) for feat in dense_features]
#     sess_feature = ['asset_num']
#
#     feats_dict = {}
#     for feat in dense_features + sparse_features + ['asset_num']:
#         feats_dict[feat] = data_tt[feat].values
#
#     session_feat = "asset_num"
#     for feat in ["sess_" + str(_x) + "_" + session_feat for _x in range(5)]:
#         feats_dict[feat] = pad_sequences(data_tt[feat], maxlen=DSIN_SESS_MAX_LEN, padding='post')
#         sparse_feature_list.append(VarLenSparseFeat(SparseFeat(feat, vocabulary_size=data_tt["asset_num"].values.max() + ID_OFFSET, embedding_name='feat'), maxlen=DSIN_SESS_MAX_LEN))
#     feats_dict["sess_length"] = data_tt["sess_length"].values
#
#     train_idx = data_tt.reset_index(drop=False)[data_tt.index<date].index.tolist()
#     test_idx = data_tt.reset_index(drop=False)[data_tt.index==date].index.tolist()
#     feats_dict_train = {}
#     feats_dict_test = {}
#     for key in feats_dict.keys():
#         feats_dict_train[key] = feats_dict[key][train_idx]
#         feats_dict_test[key] = feats_dict[key][test_idx]
#
#     param_dict = {}
#     param_dict["train_label"] = data_tt["5D_label"].values[train_idx]
#     param_dict["test_label"] = data_tt["5D_label"].values[test_idx]
#
#     param_dict["feats_dict_train"] = feats_dict_train
#     param_dict["feats_dict_test"] = feats_dict_test
#     param_dict["sess_feature"] = sess_feature
#
#     dnn_feature_columns = dense_feature_list + sparse_feature_list
#
#     param_dict["dnn_feature_columns"] = dnn_feature_columns
#     param_dict["X_col"] = dense_features + sparse_features + ['asset_num'] + ["sess_0_asset_num", "sess_1_asset_num", "sess_2_asset_num", "sess_3_asset_num", "sess_4_asset_num"]
#
#     return param_dict
#
# def get_session(_data, mode="5D"):
#     _data.sort_values(by=mode, inplace=True, ascending=False)
#     return _data.asset_num.values[0].tolist()
#
# def get_session1(idx, _data1, _len=10, session_cnt=5):
#     tmp = _data1[_data1.index<=idx].iloc[-_len-session_cnt:]
#     if len(tmp) >= _len + session_cnt:
#         tmp = tmp.sort_values(ascending=False).tolist()
#         return [tmp[x-10:x] for x in range(_len, _len + session_cnt)]
#     else:
#         return [[0 for y in range(_len)] for x in range(_len, _len + session_cnt)]
#
# def DSIN_modeling(kfold, data_tt, date, X_col, y_col, data_src, has_trained=False):
#
#     rtn_cols = ["1D", "3D", "5D", "10D", "20D"]
#
#     FRAC = 0.25
#     DIN_SESS_MAX_LEN = 50
#     DSIN_SESS_COUNT = 5
#     DSIN_SESS_MAX_LEN = 10
#     ID_OFFSET = 1000
#     SESS_COUNT = DSIN_SESS_COUNT
#     model_name = DSIN_modeling.__name__.split("_")[0]
#
#     data_tt = data_tt.fillna(0)
#
#     le = preprocessing.LabelEncoder()
#     data_tt["asset_num"] = le.fit_transform(data_tt.asset)
#     df_click_eachday = data_tt.groupby("date").apply(lambda x: get_session(x))
#     tmp_lst = pd.Series(df_click_eachday.index).apply(lambda x: get_session1(x, df_click_eachday)).tolist()
#     df_session = pd.DataFrame(tmp_lst,
#                               columns=["sess_0_asset_num", "sess_1_asset_num", "sess_2_asset_num", "sess_3_asset_num", "sess_4_asset_num"],
#                               index=df_click_eachday.index)
#     df_session["sess_length"] = [10 for x in range(len(df_session))]
#     data_tt = pd.merge(data_tt.set_index("date"), df_session, left_index=True, right_index=True, how="left")
#     params = get_feature_param(data_tt, date, ID_OFFSET, X_col)
#     params["has_trained"] = has_trained
#     train_dsin(params)


def time_series_loop(_data):

    src_dict = {
        "engi_feat": (_data.columns[2:-7].tolist(), ["3D_label"]),
        "orig_feat": (_data.columns[2:13].tolist(),["3D_label"])
    }

    for date in _data.date.drop_duplicates().sort_values().tolist()[2501:]:
        print("date:{}".format(date))
        for src_key in list(src_dict.keys()):

            X_col, y_col = src_dict[src_key]
            _data_train_valid = _data[_data.date < date].dropna().reset_index(drop=True)
            _data_test = _data[_data.date == date].dropna().reset_index(drop=True)

            if len(_data_test) == 0:
                continue

            # print("logistic start, src_key:{}".format(src_key))
            # train_and_test(_data_train_valid, _data_test, X_col, y_col, model_type="logistic")
            # print("fm start")
            # train_and_test(_data_train_valid, _data_test, X_col, y_col, model_type="fm")
            # print("dcn start")
            # train_and_test(_data_train_valid, _data_test, X_col, y_col, model_type="dcn")
            # print("xDeepFM start")
            # train_and_test(_data_train_valid, _data_test, X_col, y_col, model_type="xDeepFMWithoutEmb")

        break


if __name__ == "__main__":

    df_data = get_data()
    time_series_loop(df_data)