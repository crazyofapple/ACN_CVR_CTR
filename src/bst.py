# -*- coding:utf-8 -*-
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Input, Permute, multiply, Reshape)
from tensorflow.python.keras.regularizers import l2

from deepctr.input_embedding import create_singlefeat_inputdict, get_inputs_list, get_embedding_vec_list
from deepctr.layers.core import DNN, PredictionLayer
from src.sequence import AttentionSequencePoolingLayer, DynamicGRU, Transformer, Capsule
from deepctr.layers.utils import concat_fun, NoMask
from deepctr.utils import check_feature_config_dict


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = OrderedDict()
    for i, feat in enumerate(seq_feature_list):
        user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)

    user_behavior_length = Input(shape=(1,), name='seq_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length


def BST(feature_dim_dict, seq_feature_list, embedding_size=4, hist_len_max=16, use_bn=False, dnn_hidden_units=(200, 80),
        dnn_activation='relu', att_embedding_size=1, att_head_num=8,
        l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    check_feature_config_dict(feature_dim_dict)

    sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)
    # sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
    #                                               embeddings_initializer=RandomNormal(
    #                                                   mean=0.0, stddev=init_std, seed=seed),
    #                                               embeddings_regularizer=l2(
    #                                                   l2_reg_embedding),
    #                                               name='sparse_emb_' + str(i) + '-' + feat.name) for i, feat in
    #                          enumerate(feature_dim_dict["sparse"])}
    # print(sparse_embedding_dict)
    sparse_embedding_dict = {feat.name: Embedding(tf.cast(feat.dimension, tf.int32), embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' +
                                                       str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in seq_feature_list)) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}
    # deep_emb_list = get_embedding_vec_list(
    # deep_sparse_emb_dict, sparse_input_dict, feature_dim_dict['sparse'])
    query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                            return_feat_list=seq_feature_list)
    keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
                                           return_feat_list=seq_feature_list)
    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])

    query_emb = concat_fun(query_emb_list)
    keys_emb = concat_fun(keys_emb_list)
    print("prev: {0}".format(keys_emb))
    # hist_cap = Capsule(
    #     num_capsule=8, dim_capsule=2,
    #     routings=3, share_weights=True)(NoMask()(keys_emb))
    # print("now: {0}".format(hist_cap))
    # # exit(0)
    # # keys_emb = concat_fun(keys_emb_list)
    # hist_cap = Reshape([1, 16])(hist_cap)
    deep_input_emb = concat_fun(deep_input_emb_list)
    print("deep input emb: ", deep_input_emb)
    # print("hist_cap: ", hist_cap)
    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=False,
                                 use_positional_encoding=True, seed=seed, supports_masking=False,
                                 blinding=True)
    # print("now: {0}".format(hist))
    hists = []
    for key_emb in keys_emb_list:
        hist = Self_Attention([key_emb, key_emb, user_behavior_length, user_behavior_length])
        hists.append(hist)
    hist = concat_fun(hists)

    # Tensor("concatenate_2/concat:0", shape=(?, 50, 8), dtype=float32)
    # <tf.Tensor 'concatenate_3/concat:0' shape=(?, 4, 8) dtype=float32>
    deep_input_emb = Concatenate()([deep_input_emb, hist])
    # print(deep_input_emb)
    deep_input_emb = tf.keras.layers.Flatten()(NoMask()(deep_input_emb))
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()(
            [deep_input_emb] + list(dense_input.values()))

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, use_bn, seed)(deep_input_emb)
    final_logit = Dense(1, use_bias=False)(output)
    output = PredictionLayer(task)(final_logit)

    model_input_list = get_inputs_list(
        [sparse_input, dense_input, user_behavior_input])

    model_input_list += [user_behavior_length]

    model = tf.keras.models.Model(inputs=model_input_list, outputs=output)

    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    return model
