import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_train_test_acc2site(cfg, df, save=False, save_filename=None):
    site = cfg.test_site
    df_test = df[(df['Site'] == site)]
    df_test.reset_index(drop=True, inplace=True)
    df_train = df[(df['Site'] != site)]
    df_train.reset_index(drop=True, inplace=True)

    if save:
        df_test.to_csv(os.path.join(cfg.datapath, "Test_" + save_filename + ".csv"), index=None)
        df_train.to_csv(os.path.join(cfg.datapath, "Train_" + save_filename + ".csv"), index=None)
    return df_train, df_test


def split_cv_train_val(cfg, df, save_pickle=False, save_filename=None):
    n_fold = cfg.n_folds
    splitter = GroupKFold(n_splits=n_fold)
    splitter_iter = splitter.split(df, df['Group'], groups=df['ID'])
    cv_folds = [(train_idx, val_idx) for (train_idx, val_idx) in splitter_iter]
    train_idx_dict = {}
    val_idx_dict = {}
    for i_fold in range(n_fold):
        train_idx_fold, val_idx_fold = cv_folds[i_fold]
        train_idx_dict[i_fold] = np.array(df.index[train_idx_fold])
        val_idx_dict[i_fold] = np.array(df.index[val_idx_fold])
    if save_pickle:
        data_pickle = {}
        data_pickle['train'] = train_idx_dict
        data_pickle['validation'] = val_idx_dict
        with open(os.path.join(cfg.datapath, save_filename), 'wb') as f:
            pickle.dump(data_pickle, f)
    return train_idx_dict, val_idx_dict


def split_train_test_acc2list(cfg, df, train_list_filename, test_list_filename, save=False, save_filename=None):
    train_image_idx = pd.read_csv(os.path.join(cfg.metadatapath, train_list_filename), header=None)
    test_image_idx = pd.read_csv(os.path.join(cfg.metadatapath, test_list_filename), header=None)

    df_train = df[df['Image Index'].isin(train_image_idx[0].tolist())]
    df_test = df[df['Image Index'].isin(test_image_idx[0].tolist())]

    if save:
        df_test.to_csv(os.path.join(cfg.datapath, "Test_" + save_filename + ".csv"), index=None)
        df_train.to_csv(os.path.join(cfg.datapath, "Train_" + save_filename + ".csv"), index=None)

    return df_train, df_test
