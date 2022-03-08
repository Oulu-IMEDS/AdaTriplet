import os
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error


def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.detach().cpu().numpy()
            elif required_grad:
                x_cpu = x.cpu()
            else:
                x_cpu = x.cpu().required_grad_(False)
        elif use_numpy:
            x_cpu = x.detach().numpy()

    return x_cpu


def count_data_amounts_per_target(df, target_name):
    amount_list = []
    target_list = df[target_name].unique()

    for target in target_list:
        if target == np.nan:
            df_by_target = df[df[target_name].isnull()]
        else:
            df_by_target = df[df[target_name] == target]
        num_samples = len(df_by_target.index)
        amount_list.append(num_samples)
        # print(f'There are {num_samples} {target}-class samples.')
    return amount_list


def display_random_samples(df, root_path):
    """Function to display random samples"""
    pos = 1
    n_pairs = 3
    fig, axs = plt.subplots(len(df['Target'].unique()), n_pairs, figsize=(4, 15), sharex=True, sharey=True)

    # Loop through targets (0, 1)
    for target in range(len(df['Target'].unique())):
        # Loop through pair indices
        for pair_i in range(n_pairs):
            # Select rows by and target
            df_tmp = df[df['Target'] == target]
            # Choose a random index
            len_df = len(df_tmp.index)
            random_id = np.random.randint(0, len_df)
            # Obtain image path corresponding to the selected index
            image_path = df_tmp.iloc[random_id]['Path']
            image_fullname = os.path.join(root_path, image_path)
            # Read image
            img = cv2.imread(image_fullname, cv2.IMREAD_GRAYSCALE)
            # Show image and format its subplot
            axs[target, pair_i].imshow(img, cmap='gray')
            axs[target, pair_i].set_axis_off()
            if pair_i == 0:
                axs[target, pair_i].set_title(f'{target + 1}')
            pos += 1

    plt.tight_layout()
    plt.show()

