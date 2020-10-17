# -*- coding:utf-8 -*-
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from deepctr.input_embedding import create_singlefeat_inputdict, get_inputs_list, get_embedding_vec_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.utils import concat_fun, NoMask
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Input, Reshape)
from tensorflow.python.keras.regularizers import l2

from src.sequence import AttentionSequencePoolingLayer, Transformer, Capsule


def get_disp_loss(hisp):
    self_atten_mul = tf.matmul(hisp, tf.transpose(hisp, perm=[0, 2, 1]))
    sample_num, att_matrix_size, _ = self_atten_mul.get_shape()
    self_atten_loss = tf.square(tf.norm(self_atten_mul - np.identity(att_matrix_size.value)))
    return self_atten_loss


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = OrderedDict()
    for i, feat in enumerate(seq_feature_list):
        user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)

    user_behavior_length = Input(shape=(1,), name='seq_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length


def CapsuleNet(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=50, use_bn=False,
               dnn_hidden_units=(200, 80),
               dnn_activation='sigmoid', num_capsule=8, dim_capsule=2, routing_iterations=3, att_hidden_size=(64, 16),
               att_activation="dice", att_weight_normalization=True, att_embedding_size=1, att_head_num=8,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, alpha=1e-6, seed=1024,
               task='binary'):
    check_feature_config_dict(feature_dim_dict)

    sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)

    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' +
                                                       str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in seq_feature_list)) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}

    query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                            return_feat_list=seq_feature_list)
    keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
                                           return_feat_list=seq_feature_list)
    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])

    query_emb = concat_fun(query_emb_list)
    keys_emb = concat_fun(keys_emb_list)
    scores = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                           weight_normalization=att_weight_normalization, return_score=True)(
        [query_emb, keys_emb, user_behavior_length])

    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=True,
                                 use_positional_encoding=True, seed=seed, supports_masking=False,
                                 blinding=True)

    keys_emb = Self_Attention([keys_emb, keys_emb, user_behavior_length, user_behavior_length])

    cap = Capsule(
        num_capsule=num_capsule, dim_capsule=dim_capsule,
        routings=routing_iterations, share_weights=True, supports_masking=True)
    hist_cap = cap(keys_emb, scores=scores)
    disp_loss = get_disp_loss(hist_cap)
    hist_cap = Reshape([1, num_capsule * dim_capsule])(NoMask()(hist_cap))
    deep_input_emb = concat_fun(deep_input_emb_list)
    deep_input_emb = Concatenate()([deep_input_emb, hist_cap])

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
    model.add_loss(alpha * disp_loss)
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    return model
