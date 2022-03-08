import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import samplers
from torch.utils.data import Dataset


# import matplotlib.pyplot as plt


class DataFrameDataset(Dataset):
    """Dataset based on ``pandas.DataFrame``.

    Parameters
    ----------
    root : str
        Path to root directory of input data.
    meta_data : pandas.DataFrame
        Meta data of data and labels.
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None, which means to do nothing).
    std_size: tuple
        The size that input images are standardized to (Default: (224, 224))
    mean: tuple
        Mean to normalize image intensities (Default: (0.485, 0.456, 0.406))
    std: tuple
        Standard deviation to normalize image intensities (Default: (0.229, 0.224, 0.225))

    Raises
    ------
    TypeError
        `root` must be `str`.
    TypeError
        `meta_data` must be `pandas.DataFrame`.

    """

    def __init__(self, cfg, root, meta_data,
                 transform=None,
                 std_size=None,
                 n_channels=1,
                 mean=None,
                 std=None, form_pairs=False,
                 n_pos_pairs=2, n_neg_pairs=4,
                 df_estimated_BP=None,
                 pid_type='ID'):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(meta_data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))
        self.root = root
        self.meta_data = meta_data
        self.transform = transform
        self.std_size = std_size
        self.n_pos_pairs = n_pos_pairs
        self.n_neg_pairs = n_neg_pairs
        self.n_channels = n_channels
        self.pid_type = pid_type
        self.form_pairs = form_pairs
        self.df_estimated_BP = df_estimated_BP
        self.cfg = cfg

        if mean is None or std is None:
            if n_channels == 1:
                self.mean = [0.5]
                self.std = [0.5]
            elif n_channels == 3:
                self.mean = (0.5, 0.5, 0.5)
                self.std = (0.5, 0.5, 0.5)
            else:
                raise ValueError(f'Not supported {n_channels}')
        else:
            self.mean = mean
            self.std = std
        self.stats = {'mean': self.mean, 'std': self.std}

    def _get_img(self, path, side, transform):
        if self.n_channels == 1:
            temp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.n_channels == 3:
            temp_img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            raise ValueError(f'Not supported {self.n_channels}')

        if side == "R":
            img = cv2.flip(temp_img, 1)
        else:
            img = temp_img

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        trf_img = transform({'image': img}, return_torch=True, normalize=True, **self.stats)
        return trf_img

    def _select_knees_df(self, entry, meta_data):
        if self.pid_type == 'ID-SIDE':
            data_same_ID = meta_data[(meta_data['ID'] == entry['ID']) & (meta_data['SIDE'] == entry['SIDE'])]
        elif self.pid_type == 'ID':
            data_same_ID = meta_data[(meta_data['ID'] == entry['ID'])]
        else:
            raise ValueError(f'Not support PID field {self.pid_type}. Only support PID field as ID and ID-SIDE ')
        pos_df = data_same_ID.sample(n=self.n_pos_pairs)
        data_diff_ID = meta_data[meta_data['ID'] != entry['ID']]
        neg_df = data_diff_ID.sample(n=self.n_neg_pairs)
        pairs_df = pd.concat([pos_df, neg_df])
        # pairs_df = pairs_df.sample(frac=1).reset_index(drop=True)
        return pairs_df

    def parse_item(self, root, entry, meta_data, transform):
        raise NotImplemented

    def __getitem__(self, index):
        """Get ``index``-th parsed item of :attr:`meta_data`.

        Parameters
        ----------
        index : int
            Index of row.

        Returns
        -------
        entry : dict
            Dictionary of `index`-th parsed item.
        """
        entry = self.meta_data.iloc[index]
        entry = self.parse_item(self.root, entry, self.transform)
        if not isinstance(entry, dict):
            raise TypeError("Output of `parse_item_cb` must be `dict`, but found {}".format(type(entry)))
        return entry

    def __len__(self):
        """Get length of `meta_data`"""
        return len(self.meta_data.index)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader_each_fold(cfg, i_fold, df, index, train_transforms, valid_transforms, form_pairs=True,
                          estimated_BP=None,
                          return_dict=False):
    # Configuration
    image_path = cfg.image_crop_path
    num_workers = cfg.num_workers
    batch_size = cfg.batchsize
    pid_type = cfg.personal_id

    train_idx = index['train'][i_fold]
    val_idx = index['validation'][i_fold]

    train_df = df.iloc[train_idx].copy()
    train_df.reset_index(drop=True, inplace=True)
    val_df = df.iloc[val_idx].copy()
    val_df.reset_index(drop=True, inplace=True)

    # print("Number of train data from fold " + f'{i_fold}' + " :",
    #       sum(count_data_amounts_per_target(train_df, 'Path')))
    # print("Number of validation data from fold " + f'{i_fold}' + " :",
    #       sum(count_data_amounts_per_target(val_df, 'Path')))

    train_sampler = samplers.MPerClassSampler(train_df['PID'].tolist(), cfg.sampler.samplers_per_class,
                                              batch_size=batch_size, length_before_new_iter=len(train_df.index))
    # Dataframe to Dataset
    if cfg.data_type == 'OAI':
        mean = [127.40036 / 255.0]
        std = [78.7561 / 255.0]
        dataset = OAIDataset
    elif cfg.data_type == 'CXR':
        mean = [0.5]
        std = [0.5]
        dataset = CXRDataset

    train_ds = dataset(cfg, image_path, train_df, transform=train_transforms, n_channels=1, mean=mean, std=std)
    valid_ds = dataset(cfg, image_path, val_df, transform=valid_transforms, n_channels=1, mean=mean, std=std)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size,
                                               num_workers=num_workers, drop_last=True,
                                               sampler=train_sampler,
                                               worker_init_fn=seed_worker)
    eval_loader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False,
                                              worker_init_fn=seed_worker)
    if return_dict:
        train_loader_and_df = {'data_loader': train_loader, 'dataframe': train_df}
        eval_loader_and_df = {'data_loader': eval_loader, 'dataframe': val_df}
        return train_loader_and_df, eval_loader_and_df
    else:
        return train_loader, eval_loader


class OAIDataset(DataFrameDataset):
    """Dataset based on ``pandas.DataFrame``.

    Parameters
    ----------
    root : str
        Path to root directory of input data.
    meta_data : pandas.DataFrame
        Meta data of data and labels.
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None).
    parser_kwargs: dict
        Dict of args for :attr:`parse_item_cb` (the default is None, )

    Raises
    ------
    TypeError
        `root` must be `str`.
    TypeError
        `meta_data` must be `pandas.DataFrame`.

    """

    def __init__(self, cfg, root, meta_data,
                 transform=None,
                 std_size=None,
                 n_channels=1,
                 mean=None,
                 std=None, *args, **kwargs):
        super().__init__(cfg, root, meta_data,
                         transform,
                         std_size,
                         n_channels,
                         mean,
                         std)
        self.mean = mean
        self.std = std
        self.stats = {'mean': mean, 'std': std}

    def parse_item(self, root, entry, transform):
        img_fullname = os.path.join(root, entry['Path'])
        side = entry['SIDE']
        trf_img = self._get_img(img_fullname, side, transform)
        sample = {}
        sample['data'] = trf_img['image']
        sample['Path'] = entry['Path']
        sample['pid'] = entry['PID']

        return sample


class CXRDataset(DataFrameDataset):
    """Dataset based on ``pandas.DataFrame``.

    Parameters
    ----------
    root : str
        Path to root directory of input data.
    meta_data : pandas.DataFrame
        Meta data of data and labels.
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None).
    parser_kwargs: dict
        Dict of args for :attr:`parse_item_cb` (the default is None, )

    Raises
    ------
    TypeError
        `root` must be `str`.
    TypeError
        `meta_data` must be `pandas.DataFrame`.

    """

    def __init__(self, cfg, root, meta_data,
                 transform=None,
                 std_size=None,
                 n_channels=1,
                 mean=None,
                 std=None, *args, **kwargs):
        super().__init__(cfg, root, meta_data,
                         transform,
                         std_size,
                         n_channels,
                         mean,
                         std)
        self.mean = mean
        self.std = std
        self.stats = {'mean': mean, 'std': std}

    def parse_item(self, root, entry, transform):
        img_fullname = os.path.join(root, entry['Path'])
        side = entry['SIDE']
        trf_img = self._get_img(img_fullname, side, transform)
        sample = {}
        sample['data'] = trf_img['image']
        sample['Path'] = entry['Path']
        sample['pid'] = entry['PID']

        return sample
