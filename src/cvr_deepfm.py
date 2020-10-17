import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping
import argparse
import os
import numpy as np
import gc

try:
    from deepctr.models import DeepFM
except:
    os.system("pip install deepctr")
    os.system("pip install requests")
    os.system("pip install h5py")

from deepctr.models import DeepFM, DCN, AutoInt, AFM, NFM, PNN, xDeepFM, DIN, DIEN, DSIN
# from src.models import DeepFM
from src.baseline import Baseline
from src.bst import BST
from deepctr.utils import SingleFeat, VarLenFeat
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from src.log import create_logger


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def set_main_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument("--data-dir", type=str, default="../data/cvr_data/small_tiny_v2.csv")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/")
    parser.add_argument("--log-dir", type=str, default="../log/")
    parser.add_argument("--model", type=str, default="deepfm")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    return args


def get_train_and_feature_list(data, sparse_features, multivalue_cols, name_col='C4'):
    sparse_feature_list = [SingleFeat(feat, 1e3, hash_flag=True, dtype='float32')  # since the input is string
                           for feat in sparse_features]
    sequence_feature = []
    sequence_input = []
    sequence_input_lens = []
    for f in multivalue_cols:
        print(data.iloc[0][f])
        print(len(data.columns))
        data[f] = data[f] + "|" + data[name_col].map(str)
        print(data.iloc[0][f])
        genres_list = list(map(lambda x: list(reversed(x.split('|'))), data[f].values))
        genres_length = np.array(list(map(len, genres_list)))
        print("{0}: mean len {1}, max len {2}".format(f, np.mean(genres_length), np.max(genres_length)))
        max_len = max(genres_length)
        max_len = max(max_len, 51)
        # print(max_len)
        # Notice : padding=`post`
        genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)
        # print(genres_list)
        # sequence_feature += [VarLenFeat(f, len(key2index) + 1, max_len, 'mean')]
        sequence_feature += [VarLenFeat(f, 1e3, max_len, 'mean', hash_flag=True, dtype="string")]
        sequence_input.append(genres_list)

        sequence_input_lens.append(max_len)
    data[name_col] = data[name_col].map(float)
    sparse_input = [data[feat.name].values for feat in sparse_feature_list]

    model_input = sparse_input + sequence_input + [genres_length]
    # print("eseseswes {0}".format(sequence_input))
    return model_input, sparse_feature_list, sequence_feature, sequence_input_lens


def get_test(data, sparse_feature_list, multivalue_cols, sequence_input_lens, name_col='C4'):
    sequence_input = []

    for i, f in enumerate(multivalue_cols):
        data[f] = data[f] + "|" + data[name_col].map(str)
        genres_list = list(map(lambda x: x.split('|'), data[f].values))
        genres_length = np.array(list(map(len, genres_list)))

        genres_list = pad_sequences(genres_list, maxlen=sequence_input_lens[i], padding='post', dtype=str, value=0)
        sequence_input.append(genres_list)
    data[name_col] = data[name_col].map(float)
    sparse_input = [data[feat.name].values for feat in sparse_feature_list]

    model_input = sparse_input + sequence_input + [genres_length]

    return model_input


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is : {0} MB".format(start_mem_usg))
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: {0}".format(col))
            print("dtype before: {0}".format(props[col].dtype))

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: {0}".format(props[col].dtype))
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: {0} MB".format(mem_usg))
    print("This is {0}% of the initial size".format(100 * mem_usg / start_mem_usg))
    return props, NAlist


if __name__ == "__main__":
    args = set_main_args()
    if args.nrows == -1:
        args.nrows = None

    logger_file = os.path.join(args.log_dir, args.model + "log.txt")
    logger = create_logger(logger_file, 1)
    print = logger.info
    # old feature (wrong)
    # sparse_features = ['C126', 'C121', 'C110', 'C106', 'C101', 'C139', 'C133', 'C109', 'C130', 'C141', 'C108', 'C120',
    #                    'C132', 'C131',
    #                    'C105', 'C137', 'C112', 'C115', 'C129', 'C119', 'C113', 'C136', 'C107', 'C138', 'C103', 'C123',
    #                    'C111', 'C124',
    #                    'C128', 'C140', 'C134', 'C116', 'C125', 'C122', 'C102', 'C135', 'C117', 'C118', 'C127']
    # multivalue_cols = ['C104', 'C114']
    all_sparse_features = ['C39', 'C35', 'C34', 'C37', 'C36', 'C31', 'C30', 'C33', 'C32', 'C153', 'C152', 'C151',
                           'C150', 'C157', 'C156',
                           'C155', 'C154', 'C159', 'C158', 'C22', 'C23', 'C21', 'C27', 'C24', 'C25', 'C28', 'C168',
                           'C169', 'C166', 'C167',
                           'C164', 'C165', 'C162', 'C163', 'C160', 'C161', 'C184', 'C56', 'C55', 'C54', 'C53', 'C52',
                           'C51', 'C50', 'C188',
                           'C189', 'C59', 'C58', 'C179', 'C178', 'C141', 'C186', 'C171', 'C170', 'C173', 'C172', 'C175',
                           'C174', 'C177',
                           'C176', 'C142', 'C40', 'C191', 'C190', 'C48', 'C49', 'C121', 'C108', 'C109', 'C105', 'C106',
                           'C100', 'C101',
                           'C102', 'C103', 'C79', 'C78', 'C185', 'C71', 'C70', 'C72', 'C75', 'C74', 'C77', 'C187',
                           'C180', 'C181', 'C182',
                           'C183', 'C9', 'C8', 'C119', 'C118', 'C116', 'C1', 'C7', 'C6', 'C5', 'C4', 'C140', 'C68',
                           'C69', 'C66', 'C67',
                           'C64', 'C65', 'C62', 'C63', 'C60', 'C61', 'C144', 'C115', 'C145', 'C113', 'C112', 'C128',
                           'C129', 'C111', 'C149',
                           'C122', 'C123', 'C120', 'C110', 'C126', 'C127', 'C124', 'C125', 'C18', 'C13', 'C12', 'C11',
                           'C10', 'C17', 'C16',
                           'C15', 'C14', 'C93', 'C92', 'C91', 'C90', 'C97', 'C96', 'C95', 'C133', 'C99', 'C98', 'C139',
                           'C138', 'C57', 'C135',
                           'C132', 'C134', 'C137', 'C136', 'C131', 'C130', 'C80', 'C81', 'C82', 'C143', 'C84', 'C85',
                           'C146', 'C87', 'C88',
                           'C89', 'C94']

    sparse_features = ['C39', 'C7', 'C28', 'C5', 'C30', 'C14', 'C32', 'C18', 'C27', 'C24', 'C13', 'C9', 'C36', 'C48',
                       'C12', 'C49', 'C33', 'C8', 'C22', 'C4', 'C23', 'C16', 'C35', 'C40', 'C10', 'C6', 'C37', 'C21',
                       'C31', 'C11', 'C15', 'C34', 'C25', 'C17']

    all_multivalue_cols = ['C44', 'C45', 'C46', 'C47', 'C26', 'C41', 'C42', 'C147', 'C148', 'C73', 'C43', 'C20', 'C19',
                           'C83', 'C38', 'C104',
                           'C114']

    # multivalue_cols = ['C45', 'C26', 'C42', 'C46', 'C41', 'C44', 'C20', 'C38', 'C47', 'C43', 'C19']
    multivalue_cols = ['C38', 'C41']
    target = ['target_cvr']
    dtypes_lst = [np.float32] * 19 + [np.str] + [np.float32] * 14 + [np.str] * len(multivalue_cols) + [np.int32]
    # dtypes_lst = [col.name for name in df.dtypes]
    column_types = dict(zip(sparse_features+multivalue_cols+target, dtypes_lst))

    data = pd.read_csv(args.data_dir, sep=",", nrows=args.nrows, usecols=sparse_features+multivalue_cols+target,
                       dtype=column_types)
    # data['C4_copy'] = data['C4']
    # data.rename(columns={"C38": "C"})
    # data.info(memory_usage='deep')
    # gc.collect()
    # data, _ = reduce_mem_usage(data)
    # data.info(memory_usage='deep')
    print("-1 in clicked: {0} ".format(data['C38'].value_counts().to_dict()['-1']))
    print("-1 in converted: {0}".format(data['C41'].value_counts().to_dict()['-1']))
    # data[sparse_features] =
    # .apply(pd.to_numeric, downcast='unsigned')  # .
    # data.info(memory_usage='deep')
    data['C4_copy'] = data['C4']
    train, test = train_test_split(data, test_size=0.1)

    sparse_features += ['C4_copy']
    train_model_input, sparse_feature_list, sequence_feature, sequence_input_lens = \
        get_train_and_feature_list(train, sparse_features, multivalue_cols, name_col='C4')
    # sparse_feature_list += ['C4_copy']
    # print("#"*30)
    # # print(sparse_feature_list)
    # # print(multivalue_cols)
    # # print(sequence_input_lens)
    # print(train_model_input)
    # print("#"*30)

    test_model_input = get_test(test, sparse_feature_list, multivalue_cols, sequence_input_lens, name_col='C4')
    # model = Baseline({"sparse": sparse_feature_list, "sequence": sequence_feature}, task='binary',
    # attention_feature_name='C4')
    # model = AutoInt({"sparse": sparse_feature_list}, task='binary')
    # model = DIEN({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], 4, 50, "AUGRU", att_hidden_units=(64, 16),
    #             att_activation='sigmoid', l2_reg_embedding=1e-6, use_negsampling=False, seed=2019)
    model = BST({"sparse": sparse_feature_list}, ['C4', 'C4_copy'], 4, 51)
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    es = EarlyStopping(monitor='val_binary_crossentropy')
    csv_logger = CSVLogger(os.path.join(args.log_dir, args.model + 'log.csv'), append=True, separator=';')
    if args.mode == "train":
        history = model.fit(train_model_input, train[target].values,
                            batch_size=args.batch_size, epochs=args.epochs, verbose=2, validation_split=0.1,
                            callbacks=[csv_logger, es])
    # model.save_weights(args.save_dir + "_" + args.model + ".h5")
    else:
        assert args.mode == "test" and args.save_dir != ""

    # model.load_weights(args.save_dir + "_" + args.model + ".h5")
    pred_ans = model.predict(test_model_input, batch_size=args.batch_size)
    print("test LogLoss {0}".format(
        round(log_loss(test[target].values, np.array(pred_ans, dtype=np.float64), labels=(0, 1)), 4)))
    try:
        print("test AUC {0}".format(round(roc_auc_score(test[target].values, pred_ans), 4)))
    except Exception as e:
        pass
