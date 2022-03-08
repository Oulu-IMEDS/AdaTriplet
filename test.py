import os
import random

import hydra
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from common.data_loader import OAIDataset, CXRDataset, seed_worker
from common.evaluators import evaluate_all, evaluate_mAP_at_R
from common.image_transformation import test_transformation, test_chest_transformation
from common.main_loop import eval_loop, test_loop
from networks import BackboneModel


@hydra.main(config_path=".", config_name="config.yml")
def main(cfg):
    test_site = cfg.test_site
    seed = cfg.seed
    data_type = cfg.data_type
    preprocessed_data_filename = 'PreprocessedData_Forensic.csv'

    # Fixed Randomness
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    df_meta = pd.read_csv(os.path.join(cfg.datapath, preprocessed_data_filename))

    if data_type == 'OAI':
        filename_train_test_dataframe_postfix = f'{data_type}_{test_site}'
        df_test = pd.read_csv(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv"))
        save_each_site = f"Results_{filename_train_test_dataframe_postfix}_{cfg.personal_id}_load_{cfg.load_model_metric}_seed{cfg.seed}_query_time_{cfg.query_time}.txt"
    elif data_type == 'CXR':
        filename_train_test_dataframe_postfix = f'CXR'
        df_test = pd.read_csv(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv"))
        save_each_site = f"Results_{filename_train_test_dataframe_postfix}_{cfg.personal_id}_load_{cfg.load_model_metric}_seed{cfg.seed}_query_time_{cfg.query_time}.txt"
        if cfg.test_CXR_side == 'AP' or cfg.test_CXR_side == 'PA':
            df_test = df_test[df_test['SIDE'] == cfg.test_CXR_side]
            save_each_site = f'{cfg.test_CXR_side}_{save_each_site}'
    else:
        raise ValueError(f'Not support this {data_type} data type')

    print(f'Number test sample {df_test.shape[0]}')
    save_file_dir_sites = os.path.join(cfg.pretrained_matching_model_folder_path, save_each_site)
    file_sites = open(save_file_dir_sites, "w")

    site_mAP, site_cmc, site_mAP_at_R = do_test_reid_img(cfg, df_test)

    site_cmc_top_1 = site_cmc[0]
    site_cmc_top_5 = site_cmc[4]

    file_sites.write(f'mean AP: {site_mAP} \n')
    file_sites.write(f'mean AP at R: {site_mAP_at_R} \n')
    file_sites.write(f'CMC top 1: {site_cmc_top_1} \n')
    file_sites.write(f'CMC top 5: {site_cmc_top_5} \n')
    file_sites.close()


def do_test_reid_img(cfg, df_test):
    # Configuration
    root_path = cfg.image_crop_path
    arch_name = cfg.backbone_model
    n_fold = cfg.n_folds
    batch_size = cfg.batchsize
    device = cfg.device
    test_site = cfg.test_site
    num_workers = cfg.num_workers
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    pid = cfg.personal_id
    query_time = cfg.query_time

    if pid == 'ID':
        df_test['PID'] = df_test['ID'].astype(str)
        df_test['Encode_Visit'] = df_test['Visit'].astype(str)
    elif pid == 'ID-SIDE':
        df_test['PID'] = df_test['ID'].astype(str) + df_test['SIDE']
        df_test['Encode_Visit'] = df_test['Visit'].astype(str)
    else:
        raise ValueError(f'Not support PID field {pid}. Only support PID field as ID and ID-SIDE ')

    test_gallery = df_test[df_test['Visit'] == 0][['Path', 'PID', 'Encode_Visit']].values.tolist()
    if query_time == 'all':
        test_query = df_test[(df_test['Visit'] > 0) & (df_test['Visit'] < 156)][
            ['Path', 'PID', 'Encode_Visit']].values.tolist()
    elif query_time == '4-5':
        test_query = df_test[(df_test['Visit'] > 47) & (df_test['Visit'] < 61)][
            ['Path', 'PID', 'Encode_Visit']].values.tolist()
    elif query_time == '6-8':
        test_query = df_test[(df_test['Visit'] > 71) & (df_test['Visit'] < 97)][
            ['Path', 'PID', 'Encode_Visit']].values.tolist()
    elif query_time == '9-12':
        test_query = df_test[(df_test['Visit'] > 107) & (df_test['Visit'] < 145)][
            ['Path', 'PID', 'Encode_Visit']].values.tolist()
    else:
        test_query = df_test[df_test['Visit'] == query_time][['Path', 'PID', 'Encode_Visit']].values.tolist()
        # raise ValueError(f'Not support query time {query_time}. Only support query time by all,12,24,36,48,72,96 ')
    print(f'Query after {query_time} month(s)')

    if cfg.data_type == 'OAI':
        mean = [127.40036 / 255.0]
        std = [78.7561 / 255.0]
        # Transformation
        test_transforms = test_transformation()
        dataset = OAIDataset
    elif cfg.data_type == 'CXR':
        mean = [0.5]
        std = [0.5]
        # Transformation
        test_transforms = test_chest_transformation()
        dataset = CXRDataset

    # Training and testing and validation indices
    total_distmat = []
    total_features_query = []
    total_features_gallery = []
    for i_fold in range(n_fold):

        if cfg.data_type == 'OAI':
            pretrained_model_name = f'{arch_name}_reid_img_seed{cfg.seed}_site{test_site}_fold{i_fold}_{cfg.personal_id}_{cfg.load_model_metric}'
        elif cfg.data_type == 'CXR':
            pretrained_model_name = f'{arch_name}_reid_img_seed{cfg.seed}_fold{i_fold}_{cfg.personal_id}_{cfg.load_model_metric}'
        else:
            raise ValueError(f'Not support {cfg.data_type} data type')

        # Runing evaluation on test set
        test_ds = dataset(cfg, root_path, df_test, transform=test_transforms, n_channels=1, mean=mean, std=std)

        test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers,
                                                  worker_init_fn=seed_worker)

        # Get architecture
        model_test = BackboneModel(cfg)

        # Load trained weights
        print(pretrained_model_name)

        if cfg.pretrained_matching_model_folder_path == "None":
            output_dir = os.getcwd()
        else:
            output_dir = cfg.pretrained_matching_model_folder_path
        pretrained_model = os.path.join(output_dir, pretrained_model_name + ".pth")
        print(pretrained_model)

        checkpoint = torch.load(pretrained_model)
        model_test.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer = Adam(params=model_test.parameters(), lr=lr, weight_decay=wd)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_test = model_test.to(device)

        # Call main process
        fold_distmat, fold_features = test_loop(model_test, test_loader, test_query, test_gallery, device)
        total_distmat.append(fold_distmat)
        total_features_query.append(fold_features[0])
        total_features_gallery.append(fold_features[1])
    mean_distmat = torch.mean(torch.stack(total_distmat, 0), 0)
    mean_features_query = torch.mean(torch.stack(total_features_query, 0), 0)
    mean_features_gallery = torch.mean(torch.stack(total_features_gallery, 0), 0)
    mAP, cmc = evaluate_all(mean_distmat, query=test_query, gallery=test_gallery, return_cmc_topk=True,
                            cmc_topk=range(50))
    mAP_R = evaluate_mAP_at_R(features_query=mean_features_query, features_gallery=mean_features_gallery,
                              query=test_query, gallery=test_gallery, )
    return mAP, cmc, mAP_R

if __name__ == '__main__':
    main()