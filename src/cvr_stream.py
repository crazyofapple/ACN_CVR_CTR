from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping

matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import os as ps
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

try:
    from deepctr.models import DeepFM
except:
    ps.system("pip install deepctr")
    ps.system("pip install requests")
    ps.system("pip install h5py")

# from src.models import DeepFM

from src.capsule import CapsuleNet
from collections import OrderedDict
from deepctr.utils import SingleFeat, VarLenFeat
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from src.log import create_logger
from itertools import chain

HEADER = ["C39", "C38", "C35", "C34", "C37", "C36", "C31", "C30", "C33", "C32", "C153", "C152", "C151", "C150", "C157",
          "C156", "C155", "C154", "C159", "C158", "C22", "C23", "C20", "C21", "C26", "C27", "C24", "C25", "C28", "C168",
          "C169", "C166", "C167", "C164", "C165", "C162", "C163", "C160", "C161", "C184", "C57", "C56", "C55", "C54",
          "C53", "C52", "C51", "C50", "C188", "C189", "C59", "C58", "C179", "C178", "C141", "C133", "C186", "C171",
          "C170", "C173", "C172", "C175", "C174", "C177", "C176", "C191", "C44", "C45", "C46", "C47", "C40", "C41",
          "C42", "C43", "C48", "C49", "C190", "C121", "C108", "C109", "C104", "C105", "C106", "C100", "C101", "C102",
          "C103", "C79", "C78", "C185", "C71", "C70", "C73", "C72", "C75", "C74", "C77", "C187", "C180", "C181", "C182",
          "C183", "C9", "C8", "C119", "C118", "C116", "C1", "C114", "C7", "C6", "C5", "C4", "C140", "C68", "C69", "C66",
          "C67", "C64", "C65", "C62", "C63", "C60", "C61", "C143", "C144", "C115", "C145", "C113", "C147", "C112",
          "C148", "C128", "C129", "C111", "C149", "C122", "C123", "C120", "C110", "C126", "C127", "C124", "C125", "C19",
          "C18", "C13", "C12", "C11", "C10", "C17", "C16", "C15", "C14", "C93", "C92", "C91", "C90", "C97", "C96",
          "C95", "C94", "C99", "C98", "C139", "C138", "C135", "C134", "C137", "C136", "C131", "C130", "C80", "C81",
          "C82", "C83", "C84", "C85", "C146", "C87", "C88", "C89", "C132", "C142", "target_cvr", "target_ctr", "id",
          "type_label"]
# HEADER = ['C39', 'C38', 'C35', 'C34', 'C37', 'C36', 'C31', 'C30', 'C33', 'C32', 'C153', 'C152', 'C151', 'C150',
#                    'C157', 'C156', 'C155', 'C154', 'C159', 'C158', 'C22', 'C23', 'C20', 'C21', 'C26', 'C27', 'C24',
#                    'C25', 'C28', 'C168', 'C169', 'C166', 'C167', 'C164', 'C165', 'C162', 'C163', 'C160', 'C161', 'C184',
#                    'C57', 'C56', 'C55', 'C54', 'C53', 'C52', 'C51', 'C50', 'C188', 'C189', 'C59', 'C58', 'C179', 'C178',
#                    'C141', 'C133', 'C186', 'C171', 'C170', 'C173', 'C172', 'C175', 'C174', 'C177', 'C176', 'C191',
#                    'C44', 'C45', 'C46', 'C47', 'C40', 'C41', 'C42', 'C43', 'C48', 'C49', 'C190', 'C121', 'C108', 'C109',
#                    'C104', 'C105', 'C106', 'C100', 'C101', 'C102', 'C103', 'C79', 'C78', 'C185', 'C71', 'C70', 'C73',
#                    'C72', 'C75', 'C74', 'C77', 'C187', 'C180', 'C181', 'C182', 'C183', 'C9', 'C8', 'C119', 'C118', 'C3',
#                    'C2', 'C1', 'C114', 'C7', 'C6', 'C5', 'C4', 'C140', 'C68', 'C69', 'C66', 'C67', 'C64', 'C65', 'C62',
#                    'C63', 'C60', 'C61', 'C143', 'C116', 'C144', 'C115', 'C145', 'C113', 'C147', 'C112', 'C148', 'C128',
#                    'C129', 'C111', 'C149', 'C122', 'C123', 'C120', 'C110', 'C126', 'C127', 'C124', 'C125', 'C19', 'C18',
#                    'C13', 'C12', 'C11', 'C10', 'C17', 'C16', 'C15', 'C14', 'C93', 'C92', 'C91', 'C90', 'C97', 'C96',
#                    'C95', 'C94', 'C99', 'C98', 'C139', 'C138', 'C135', 'C134', 'C137', 'C136', 'C131', 'C130', 'C80',
#                    'C81', 'C82', 'C83', 'C84', 'C85', 'C146', 'C87', 'C88', 'C89', 'C132', 'C142', 'target_cvr',
#                    'target_ctr', 'id', 'type_label'] # online used
sparse_features = ['C39', 'C7', 'C28', 'C5', 'C30', 'C14', 'C32', 'C18', 'C27', 'C24', 'C13', 'C9', 'C36', 'C48',
                   'C12', 'C49', 'C33', 'C8', 'C22', 'C4', 'C23', 'C16', 'C35', 'C40', 'C10', 'C6', 'C37', 'C21',
                   'C31', 'C11', 'C15', 'C34', 'C25', 'C17', 'C4_copy']
multivalue_cols = ['C38', 'C41']
target = ['target_cvr']
test_ys = []


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.binary_crossentropy = {'batch': [], 'epoch': []}
        self.auc = {'batch': [], 'epoch': []}
        self.val_binary_crossentropy = {'batch': [], 'epoch': []}
        self.val_auc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.binary_crossentropy['batch'].append(logs.get('binary_crossentropy'))
        self.auc['batch'].append(logs.get('auc'))
        self.val_binary_crossentropy['batch'].append(logs.get('val_binary_crossentropy'))
        self.val_auc['batch'].append(logs.get('val_auc'))

    def on_epoch_end(self, batch, logs={}):
        self.binary_crossentropy['epoch'].append(logs.get('binary_crossentropy'))
        self.auc['epoch'].append(logs.get('auc'))
        self.val_binary_crossentropy['epoch'].append(logs.get('val_binary_crossentropy'))
        self.val_auc['epoch'].append(logs.get('val_auc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.binary_crossentropy[loss_type]))
        plt.figure()
        # auc
        plt.plot(iters, self.auc[loss_type], 'r', label='train auc')
        # loss
        plt.plot(iters, self.binary_crossentropy[loss_type], 'b', label='train logloss')
        if loss_type == 'epoch':
            # val_auc
            plt.plot(iters, self.val_auc[loss_type], 'b', label='val auc')
            # val_loss
            plt.plot(iters, self.val_binary_crossentropy[loss_type], 'k', label='val logloss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('auc-logloss')
        plt.legend(loc="upper right")
        plt.title('Model ACN-T')
        plt.savefig('auc_and_logloss.pdf')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def set_main_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument("--train-dir", type=str, default="../data/cvr_data/small_tiny_v2.txt")
    parser.add_argument("--test-dir", type=str, default="../data/cvr_data/small_tiny_v2.txt")
    parser.add_argument("--val-dir", type=str, default="../data/cvr_data/small_tiny_v2.txt")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/")
    parser.add_argument("--log-dir", type=str, default="../log/")
    parser.add_argument("--model", type=str, default="deepfm")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-size", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=4)
    args = parser.parse_args()
    return args


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def get_feature_list(sparse_features, multivalue_cols, max_len=50):
    sparse_feature_list = [SingleFeat(feat, 1000, hash_flag=True, dtype='string')  # since the input is string
                           for feat in sparse_features]
    sequence_feature = []
    for f in multivalue_cols:
        sequence_feature += [VarLenFeat(f, 1000, max_len, 'mean', hash_flag=True, dtype="string")]
    return sparse_feature_list, sequence_feature


def generate_arrays_from_file(path, phrase, max_len=50, name_col='C4', use_neg=False, ads_dir=None):
    if phrase == "train":
        length = args.train_size
    elif phrase == "val":
        length = args.val_size
    else:
        length = args.test_size

    sparse_inputs = OrderedDict()
    sequence_inputs = OrderedDict()
    sequence_inputs_len = OrderedDict()
    y_inputs = []

    while True:
        with open(path) as f:
            for i, line in enumerate(f):
                i = i + 1
                if i > length:
                    break
                data = line.strip().split(",")

                for fea in sparse_features:
                    if fea not in sparse_inputs:
                        sparse_inputs[fea] = []
                    if fea[-5:] == "_copy":
                        sparse_inputs[fea].append(data[HEADER.index(fea[:-5])])
                    else:
                        sparse_inputs[fea].append(data[HEADER.index(fea)])
                    if sparse_inputs[fea][-1] == "-1":
                        sparse_inputs[fea][-1] = "1"

                for fea in multivalue_cols:
                    if fea not in sequence_inputs:
                        sequence_inputs[fea] = []
                    genres_list = list(reversed(data[HEADER.index(fea)].split('|')))  # + [data[HEADER.index(name_col)]]
                    genres_list = [x if x != "-1" else "1" for x in genres_list]
                    # print("fea: {0}, value: {1}".format(fea, genres_list))
                    sequence_inputs[fea].append(genres_list)
                    if fea not in sequence_inputs_len:
                        sequence_inputs_len[fea] = []

                    sequence_inputs_len[fea].append(len(genres_list))
                y_inputs.append(int(data[HEADER.index(target[0])]))

                if i % args.batch_size == 0 or i == length:
                    inputs = [np.array(x) for x in list(sparse_inputs.values())]
                    for x in sequence_inputs.values():
                        genres_list = pad_sequences(x, maxlen=max_len, padding='post', dtype=str, value=0)
                        inputs.append(genres_list)
                    if use_neg:
                        assert ads_dir is not None

                    for x, y in sequence_inputs_len.items():
                        inputs.append(np.array(y))
                        break
                    outputs = np.array(y_inputs)

                    sparse_inputs = OrderedDict()
                    sequence_inputs = OrderedDict()
                    sequence_inputs_len = OrderedDict()
                    y_inputs = []
                    yield inputs, outputs


def generate_arrays_from_test_file(path):
    global test_ys
    test_ys = []

    with open(path) as f:
        for i, line in enumerate(f):
            if i == args.test_size:
                break
            data = line.strip().split(",")
            test_ys.append(int(data[HEADER.index(target[0])]))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


if __name__ == "__main__":
    args = set_main_args()
    if args.nrows == -1:
        args.nrows = None

    logger_file = ps.path.join(args.log_dir, args.model + "log.txt")
    logger = create_logger(logger_file, 1)
    print = logger.info

    sparse_feature_list, sequence_feature = get_feature_list(sparse_features, multivalue_cols)

    # print("#" * 30)
    # print(sparse_feature_list)
    # print(multivalue_cols)
    # print("#" * 30)

    # model = DeepFM({"sparse": sparse_feature_list, "sequence": sequence_feature}, task='binary')
    # model = DIN({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], task='binary', hist_len_max=50)
    # # model = DIEN({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], 4, 50, "AUGRU", att_hidden_units=(64, 16),
    # #             att_activation='sigmoid', l2_reg_embedding=1e-6, use_negsampling=True, seed=2019)
    # model = BST({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], 4, 51, att_embedding_size=1, att_head_num=4,)
    model = CapsuleNet({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], 5, 50, dnn_hidden_units=(135, 67),
                       att_embedding_size=2, att_head_num=5, att_activation="sigmoid", alpha=0, num_capsule=5)
    # exit(0)
    history = LossHistory()
    model.compile("adagrad", "binary_crossentropy", metrics=['binary_crossentropy', auc], )
    # model.compile("adagrad", loss=[focal_loss(alpha=.25, gamma=2)], metrics=['binary_crossentropy', auc], )

    es = EarlyStopping(monitor='val_binary_crossentropy')
    csv_logger = CSVLogger(ps.path.join(args.log_dir, args.model + 'log.csv'), append=True, separator=';')
    if args.mode == "train":
        es = EarlyStopping(monitor='val_binary_crossentropy')
        hhistory = model.fit_generator(generate_arrays_from_file(args.train_dir, phrase="train"),
                                       steps_per_epoch=(args.train_size + args.batch_size - 1) // args.batch_size,
                                       epochs=args.epochs, verbose=2,
                                       validation_data=generate_arrays_from_file(args.val_dir, phrase="val"),
                                       validation_steps=(args.val_size + args.batch_size - 1) // args.batch_size,
                                       callbacks=[csv_logger, es])
        # print(hhistory.history.keys())

        # history.loss_plot('batch')
    # model.save_weights(args.save_dir + "_" + args.model + ".h5")
    else:
        assert args.mode == "test" and args.save_dir != ""

    # model.load_weights(args.save_dir + "_" + args.model + ".h5")

    pred_ans = model.predict_generator(generate_arrays_from_file(args.test_dir, phrase="test"),
                                       (args.test_size + args.batch_size - 1) // args.batch_size, )
    # print(pred_ans)
    generate_arrays_from_test_file(args.test_dir)
    print("test LogLoss {0}".format(
        round(log_loss(test_ys, np.array(pred_ans, dtype=np.float64), labels=(0, 1)), 4)))
    try:
        print("test AUC {0}".format(round(roc_auc_score(test_ys, pred_ans), 4)))
    except Exception as e:
        pass