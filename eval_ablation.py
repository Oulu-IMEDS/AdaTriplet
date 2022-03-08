import pickle
import random

import hydra
import numpy as np
import os
import pandas as pd
import torch
from torch.optim import Adam
from common.dist_metric import DistanceMetric
from common.evaluators import Evaluator

from common.data_loader import data_loader_each_fold
from common.image_transformation import train_transformation, test_transformation, train_chest_transformation, \
    test_chest_transformation
from models.networks import BackboneModel
from train import do_train_reid_img


# Data Loader
@hydra.main(config_path=".", config_name="config.yml")
def main(cfg):
    test_site = cfg.test_site
    n_folds = cfg.n_folds
    cutoff_age_min = cfg.cutoff_age_min
    cutoff_age_max = cfg.cutoff_age_max
    KL_grade_train = cfg.KL.KL_grade_train
    KL_grade_test = cfg.KL.KL_grade_test
    preprocessed_data_filename = 'PreprocessedData_Forensic.csv'
    cfg.preprocess_data = preprocessed_data_filename

    if cfg.data_type == 'OAI':
        filename_train_test_dataframe_postfix = f'{test_site}_age{cutoff_age_min}{cutoff_age_max}_train-KL{KL_grade_train}_test-KL{KL_grade_test}'
        save_pickle_filename = f'train_validation_index_pickle_{n_folds}folds' \
                               f'_site{test_site}_random_seed_{cfg.seed}.p'
        df_site_leftover = pd.read_csv(
            os.path.join(cfg.datapath, f"Train_not_include_site_{filename_train_test_dataframe_postfix}.csv"))
    elif cfg.data_type == 'CXR':
        filename_train_test_dataframe_postfix = f'CXR'
        save_pickle_filename = f'train_validation_index_pickle_{n_folds}folds' \
                               f'_random_seed_{cfg.seed}.p'
        df_site_leftover = pd.read_csv(
            os.path.join(cfg.datapath, f"Train_{filename_train_test_dataframe_postfix}.csv"))
    else:
        raise ValueError(f'Not support {cfg.data_type} data type')


    # Fixed Randomness
    pd.options.mode.chained_assignment = None
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    index = pickle.load(open(os.path.join(cfg.datapath, save_pickle_filename), "rb"))

    df_meta = df_site_leftover

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Configuration
    arch_name = cfg.backbone_model
    i_fold = cfg.i_fold
    device = cfg.device
    test_site = cfg.test_site
    pid = cfg.personal_id
    query_time = cfg.query_time

    if pid == 'ID':
        ids = df_meta["ID"].unique().tolist()
        map_ids_idx = {id: index for index, id in enumerate(ids)}
        indices = [map_ids_idx[row['ID']] for _, row in df_meta.iterrows()]
        df_meta["PID"] = indices
        df_meta['Encode_Visit'] = df_meta['Visit'].astype(str) + df_meta['SIDE']
    elif pid == 'ID-SIDE':
        df_meta["ID_encode"] = df_meta['ID'].astype(str) + df_meta['SIDE']
        ids = df_meta["ID_encode"].unique().tolist()
        map_ids_idx = {id: index for index, id in enumerate(ids)}
        indices = [map_ids_idx[row['ID_encode']] for _, row in df_meta.iterrows()]
        df_meta["PID"] = indices
        df_meta['Encode_Visit'] = df_meta['Visit'].astype(str)
    else:
        raise ValueError(f'Not support PID field {pid}. Only support PID field as ID and ID-SIDE ')

    # Transformation
    if cfg.data_type == 'OAI':
        train_transforms = train_transformation()
        valid_transforms = test_transformation()
        saved_model_name = f'{arch_name}_reid_img_seed{cfg.seed}_site{test_site}_fold{i_fold}'
    elif cfg.data_type == 'CXR':
        train_transforms = train_chest_transformation()
        valid_transforms = test_chest_transformation()
        saved_model_name = f'{arch_name}_reid_img_seed{cfg.seed}_fold{i_fold}'
    else:
        raise ValueError(f'Not support {cfg.data_type} data type')


    print(saved_model_name)
    train_dict, eval_dict = data_loader_each_fold(cfg, i_fold, df_meta, index, train_transforms,
                                                  valid_transforms, return_dict=True)

    # Architecture
    model = BackboneModel(cfg)

    # Load trained weights
    pretrained_model_name = f'{saved_model_name}_{cfg.personal_id}_mAP'
    print(pretrained_model_name)

    if cfg.pretrained_matching_model_folder_path == "None":
        output_dir = os.getcwd()
    else:
        output_dir = cfg.pretrained_matching_model_folder_path
    pretrained_model = os.path.join(output_dir, pretrained_model_name + ".pth")
    print(pretrained_model)

    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to(device)

    eval_loader = eval_dict['data_loader']
    eval_df = eval_dict['dataframe']
    val_gallery = eval_df[eval_df['Visit'] == 0][['Path', 'PID', 'Encode_Visit']].values.tolist()
    if query_time == 'all':
        val_query = eval_df[eval_df['Visit'] > 0][['Path', 'PID', 'Encode_Visit']].values.tolist()
    else:
        val_query = eval_df[eval_df['Visit'] >= query_time][['Path', 'PID', 'Encode_Visit']].values.tolist()



    with torch.no_grad():
        metric = DistanceMetric(algorithm=cfg.eval_dismat_algorithm)
        evaluator = Evaluator(model, 0, device)
        mAP, cmc_0 = evaluator.evaluate(eval_loader, val_query, val_gallery, metric)

    if cfg.data_type == 'OAI':
        save_filename = f"Validation_metrics_case_seed{cfg.seed}_site{cfg.test_site}_fold{cfg.i_fold}_{cfg.query_time}.txt"
    elif cfg.data_type == 'CXR':
        save_filename = f"Validation_metrics_case_seed{cfg.seed}_fold{cfg.i_fold}{cfg.query_time}.txt"

    save_file_dir = os.path.join(output_dir, save_filename)
    file = open(save_file_dir, "w")
    file.write(f'mean AP: {mAP} \n')
    file.write(f'CMC top 1 : {cmc_0} \n')
    file.close()


if __name__ == '__main__':
    main()
