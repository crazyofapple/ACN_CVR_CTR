# -*- coding:utf-8 -*-

import tensorflow as tf

from src.input_embedding import preprocess_input_embedding, get_linear_logit
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict


def Baseline(feature_dim_dict, attention_feature_name=None, with_linear=False, embedding_size=5, dnn_hidden_units=(135, 67), l2_reg_linear=0.00001,
            l2_reg_embedding=0.00001, l2_reg_dnn=0,
           init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the Baseline Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like
           {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = \
        preprocess_input_embedding(feature_dim_dict,
                                   embedding_size,
                                   l2_reg_embedding,
                                   l2_reg_linear, init_std,
                                   seed,
                                   create_linear_weight=True,
                                   use_var_attention=(
                                       True if attention_feature_name else False),
                                   attention_feature_name=attention_feature_name)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)

    deep_input = concat_fun(deep_emb_list, axis=1)
    deep_input = tf.keras.layers.Flatten()(deep_input)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                   dnn_use_bn, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)

    if len(dnn_hidden_units) == 0:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) > 0 and with_linear:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(dnn_hidden_units) > 0 and not with_linear:
        final_logit = deep_logit
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
