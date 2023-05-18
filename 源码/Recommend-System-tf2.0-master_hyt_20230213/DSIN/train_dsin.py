# coding: utf-8
import os

import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


FRAC = 0.25
DIN_SESS_MAX_LEN = 50
DSIN_SESS_COUNT = 10
DSIN_SESS_MAX_LEN = 20
ID_OFFSET = 1000

from deepctr.models import DSIN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))


def train_dsin(param_dict):

    feats_dict_train = param_dict["feats_dict_train"]
    train_label = param_dict["train_label"]
    feats_dict_test = param_dict["feats_dict_test"]
    test_label = param_dict["test_label"]

    sess_count = param_dict["DSIN_SESS_COUNT"]
    sess_feature_list = ["asset_num"]

    dnn_feature_columns = param_dict["dnn_feature_columns"]

    _model = DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=sess_count, bias_encoding=False,
                 att_embedding_size=4, att_head_num=1, dnn_hidden_units=(400, 160), dnn_activation='relu',
                 dnn_dropout=0.5)
    _model.compile('adagrad', 'binary_crossentropy', metrics=['binary_crossentropy', ])

    BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    _model.fit(feats_dict_train, train_label, batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1)

    pred_ans = _model.predict(feats_dict_test, TEST_BATCH_SIZE)
    train_ans = _model.predict(feats_dict_train, TEST_BATCH_SIZE)

    print("test LogLoss", round(log_loss(test_label, pred_ans), 4),
          "train AUC", round(roc_auc_score(train_label, train_ans), 4),
          "test AUC", round(roc_auc_score(test_label, pred_ans), 4),

          )