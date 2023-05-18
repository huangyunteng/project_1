import os.path

from Logistic.model import Logistic
from FM.model import FM
from DCN.model import DCNWithOutEmb
from xDeepFM.model import xDeepFMWithoutEmb

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.decomposition import PCA

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
# def model_value(model, X_valid, y_valid):
#     # 评估
#     pre = model(X_valid)
#     pre = [1 if x > 0.5 else 0 for x in pre]
#     # AUC = accuracy_score(y_valid, pre)
#     AUC = round(roc_auc_score(y_valid, pre), 4)
#     # print("AUC:{}".format(round(AUC, 5)))
#     return AUC


def train_logistic(X_train, y_train):
    k = 8
    w_reg = 1e-4
    model = Logistic(k, w_reg)
    optimizer = optimizers.SGD(0.01)
    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(40):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    # print("loss:{}".format(round(loss.numpy(), 5)))
    return model, loss.numpy()


def train_FM(X_train, y_train):

    k = 8
    w_reg = 1e-4
    v_reg = 1e-4

    model = FM(k, w_reg, v_reg)
    optimizer = optimizers.SGD(0.01)
    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    # print("loss:{}".format(round(loss.numpy(), 5)))
    return model, loss.numpy()


def train_DCN(X_train, y_train):

    hidden_units = [256, 128, 64]
    model = DCNWithOutEmb(hidden_units, 1, activation='relu', layer_num=6)
    optimizer = optimizers.SGD(0.01)

    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    return model, loss.numpy()


def train_xDeepFMWithoutEmb(X_train_tt, y_train_tt):

    hidden_units = [256, 128, 64]
    dropout = 0.3
    cin_size = [128, 128]
    model = xDeepFMWithoutEmb(cin_size, hidden_units, dropout=dropout)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tt, y_train_tt))
    train_dataset = train_dataset.batch(256).prefetch(tf.data.experimental.AUTOTUNE)

    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for epoch in range(5):
        loss_summary = []
        for batch, data_batch in enumerate(train_dataset):
            X_train, y_train = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                y_pre = model(X_train)
                loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
            if batch%20==0:
                print('epoch: {} batch: {} loss: {}'.format(epoch, batch, loss.numpy()))
            loss_summary.append(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", np.mean(loss_summary), step=epoch)
    return model, loss.numpy()


def logistic_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src):

    model_name = logistic_modeling.__name__.split("_")[0]

    _loss_lst = []
    auc_train_lst = []
    for n_fold, (tr_idx, val_idx) in enumerate(kfold.split(_data_train_valid[X_col], _data_train_valid[y_col])):
        tr_x, tr_y, val_x, val_y = _data_train_valid[X_col].iloc[tr_idx], _data_train_valid[y_col].iloc[tr_idx], \
                                   _data_train_valid[X_col].iloc[val_idx], _data_train_valid[y_col].iloc[val_idx]

        model, _loss = train_logistic(tr_x.values, tr_y.values)
        auc_train = model_value(model, val_x.values, val_y.values)

        _loss_lst.append(round(_loss, 4))
        auc_train_lst.append(round(auc_train, 4))

    model, test_loss = train_logistic(_data_train_valid[X_col].values, _data_train_valid[y_col].values)
    auc_test = model_value(model, _data_test[X_col].values, _data_test[y_col].values)
    params = pd.Series([x[0] for x in model.weights[1].numpy()], index=X_col).sort_values(ascending=False)
    params.reset_index(drop=False).to_csv("params_" + model_name + "_" + data_src + ".csv", index=False)

    df_evalua = pd.DataFrame(
        index=["loss_tr_l", "loss_tr_m", "loss_tr_std", "auc_tr_l", "auc_tr_m", "auc_tr_s", "loss_te", "auc_te"])
    df_evalua[model_name] = [_loss_lst, np.mean(_loss_lst), np.std(_loss_lst), auc_train_lst, np.mean(auc_train_lst),
                             np.std(auc_train_lst), test_loss, auc_test]
    df_evalua.reset_index(drop=False).to_csv("df_evalua_" + model_name + "_" + data_src + ".csv", index=False)

    print("df_evalua of {} in date {}: {}".format(data_src, str(_data_test.date.values[0]).split("T")[0], df_evalua))


def FM_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src):

    model_name = FM_modeling.__name__.split("_")[0]

    _loss_lst = []
    auc_train_lst = []
    for n_fold, (tr_idx, val_idx) in enumerate(kfold.split(_data_train_valid[X_col], _data_train_valid[y_col])):
        tr_x, tr_y, val_x, val_y = _data_train_valid[X_col].iloc[tr_idx], _data_train_valid[y_col].iloc[tr_idx], \
                                   _data_train_valid[X_col].iloc[val_idx], _data_train_valid[y_col].iloc[val_idx]

        model, _loss = train_FM(tr_x.values, tr_y.values)
        auc_train = model_value(model, val_x.values, val_y.values)

        _loss_lst.append(round(_loss, 4))
        auc_train_lst.append(round(auc_train, 4))

    model, test_loss = train_FM(_data_train_valid[X_col].values, _data_train_valid[y_col].values)
    auc_test = model_value(model, _data_test[X_col].values, _data_test[y_col].values)
    params = pd.Series([x[0] for x in model.weights[1].numpy()], index=X_col).sort_values(ascending=False)
    params.reset_index(drop=False).to_csv("params_" + model_name + "_" + data_src + ".csv", index=False)

    df_evalua = pd.DataFrame(index=["loss_tr_l","loss_tr_m","loss_tr_std","auc_tr_l","auc_tr_m","auc_tr_s","loss_te","auc_te"])
    df_evalua[model_name] = [_loss_lst, np.mean(_loss_lst), np.std(_loss_lst), auc_train_lst, np.mean(auc_train_lst), np.std(auc_train_lst), test_loss, auc_test]
    df_evalua.reset_index(drop=False).to_csv("df_evalua_" + model_name + "_" + data_src + ".csv", index=False)

    print("df_evalua of {} in date {}: {}".format(data_src, str(_data_test.date.values[0]).split("T")[0], df_evalua))


def DCN_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src):

    model_name = DCN_modeling.__name__.split("_")[0]

    _loss_lst = []
    auc_train_lst = []
    for n_fold, (tr_idx, val_idx) in enumerate(kfold.split(_data_train_valid[X_col], _data_train_valid[y_col])):
        tr_x, tr_y, val_x, val_y = _data_train_valid[X_col].iloc[tr_idx], _data_train_valid[y_col].iloc[tr_idx], \
                                   _data_train_valid[X_col].iloc[val_idx], _data_train_valid[y_col].iloc[val_idx]

        model, _loss = train_DCN(tr_x.values, tr_y.values)
        auc_train = model_value(model, val_x.values, val_y.values)

        _loss_lst.append(round(_loss, 4))
        auc_train_lst.append(round(auc_train, 4))

    model, test_loss = train_DCN(_data_train_valid[X_col].values, _data_train_valid[y_col].values)
    auc_test = model_value(model, _data_test[X_col].values, _data_test[y_col].values)
    # params = pd.Series([x[0] for x in model.weights[1].numpy()], index=X_col).sort_values(ascending=False)
    # params.reset_index(drop=False).to_csv("params_" + model_name + "_" + data_src + ".csv", index=False)

    df_evalua = pd.DataFrame(index=["loss_tr_l","loss_tr_m","loss_tr_std","auc_tr_l","auc_tr_m","auc_tr_s","loss_te","auc_te"])
    df_evalua[model_name] = [_loss_lst, np.mean(_loss_lst), np.std(_loss_lst), auc_train_lst, np.mean(auc_train_lst), np.std(auc_train_lst), test_loss, auc_test]
    df_evalua.reset_index(drop=False).to_csv("df_evalua_" + model_name + "_" + data_src + ".csv", index=False)

    print("df_evalua of {} in date {}: {}".format(data_src, str(_data_test.date.values[0]).split("T")[0], df_evalua))


def xDeepFM_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src):

    model_name = xDeepFM_modeling.__name__.split("_")[0]

    _loss_lst = []
    auc_train_lst = []
    for n_fold, (tr_idx, val_idx) in enumerate(kfold.split(_data_train_valid[X_col], _data_train_valid[y_col])):
        tr_x, tr_y, val_x, val_y = _data_train_valid[X_col].iloc[tr_idx], _data_train_valid[y_col].iloc[tr_idx], \
                                   _data_train_valid[X_col].iloc[val_idx], _data_train_valid[y_col].iloc[val_idx]

        model, _loss = train_xDeepFMWithoutEmb(tr_x.values, tr_y.values)
        auc_train = model_value(model, val_x.values, val_y.values)

        _loss_lst.append(round(_loss, 4))
        auc_train_lst.append(round(auc_train, 4))

    model, test_loss = train_xDeepFMWithoutEmb(_data_train_valid[X_col].values, _data_train_valid[y_col].values)
    auc_test = model_value(model, _data_test[X_col].values, _data_test[y_col].values)

    df_evalua = pd.DataFrame(index=["loss_tr_l","loss_tr_m","loss_tr_std","auc_tr_l","auc_tr_m","auc_tr_s","loss_te","auc_te"])
    df_evalua[model_name] = [_loss_lst, np.mean(_loss_lst), np.std(_loss_lst), auc_train_lst, np.mean(auc_train_lst), np.std(auc_train_lst), test_loss, auc_test]
    df_evalua.reset_index(drop=False).to_csv("df_evalua_" + model_name + "_" + data_src + ".csv", index=False)

    print("df_evalua of {} in date {}: {}".format(data_src, (_data_test.date.values[0]).split("T")[0], df_evalua))
#
# from sklearn import preprocessing
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from DSIN.train_dsin import train_dsin
# from tqdm import tqdm
# from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
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


from sklearn.model_selection import StratifiedKFold

def time_series_loop(_data):

    src_dict = {
        "engi_feat": (_data.columns[2:-7].tolist(), ["5D_label"]),
        "orig_feat": (_data.columns[2:13].tolist(),["5D_label"])
    }

    has_trained = False
    for date in _data.date.drop_duplicates().sort_values().tolist()[2501:]:
        print("date:{}".format(date))
        for src_key in list(src_dict.keys()):

            data_src = src_key.split("_")[0]
            X_col, y_col = src_dict[src_key]

            _data_train_valid = _data[_data.date < date].dropna().reset_index(drop=True)
            _data_test = _data[_data.date == date].dropna().reset_index(drop=True)

            if len(_data_test) == 0:
                continue
            kfold = StratifiedKFold(n_splits=2, shuffle=False, random_state=None)

            # 逻辑回归
            logistic_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src)

            # FM二阶交叉
            FM_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src)

            # DCN
            DCN_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src)

            # # xDeepFM
            # xDeepFM_modeling(kfold, _data_train_valid, _data_test, X_col, y_col, data_src)

            # # DSIN
            # DSIN_modeling(kfold, _data, date, X_col, y_col, data_src, has_trained)
            # has_trained = True






        break





if __name__ == "__main__":

    df_data = get_data()
    time_series_loop(df_data)
