import os
import pickle
import random

import hydra
import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import distances, reducers, losses, miners
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, CosineEmbeddingLoss, BCELoss
from torch.optim import Adam

from common.data_loader import data_loader_each_fold
from common.image_transformation import train_transformation, test_transformation, train_chest_transformation, \
    test_chest_transformation
from common.losses.losses import TripletCustomMarginLoss, HingeLoss, LowerBoundLoss, LogSumExpLoss
from common.main_loop import eval_loop, train_loop
from common.miners.triplet_automargin_miner import TripletAutoMarginMiner, TripletAdaptiveMiner, TripletSCTMiner, \
    TripletAutoParamsMiner
from common.miners.triplet_margin_miner import TripletMarginMiner
from common.prepare_dataframe import create_df_OAI_forensic, create_df_CXR_forensic
from common.split_train_test_data import split_train_test_acc2site, split_cv_train_val, split_train_test_acc2list
from networks import BackboneModel


@hydra.main(config_path=".", config_name="config.yml")
def main(cfg):
    test_site = cfg.test_site
    n_folds = cfg.n_folds
    data_type = cfg.data_type
    preprocessed_data_filename = 'PreprocessedData_Forensic.csv'
    cfg.preprocess_data = preprocessed_data_filename

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

    if data_type == 'OAI':
        print(f'Choosing test site {test_site}')
        filename_train_test_dataframe_postfix = f'{data_type}_{test_site}'
        save_pickle_filename = f'{data_type}_train_validation_index_pickle_{n_folds}folds' \
                               f'_site{test_site}_random_seed_{cfg.seed}.p'

        if not os.path.isfile(os.path.join(cfg.datapath, preprocessed_data_filename)):
            df_meta = create_df_OAI_forensic(cfg, applied_kl=True, save=True,
                                             filename=preprocessed_data_filename)
        else:
            df_meta = pd.read_csv(os.path.join(cfg.datapath, preprocessed_data_filename))

        if not os.path.isfile(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv")):
            df_train, df_test = split_train_test_acc2site(cfg, df_meta, save=True,
                                                          save_filename=filename_train_test_dataframe_postfix)
        else:
            df_test = pd.read_csv(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv"))
            df_train = pd.read_csv(
                os.path.join(cfg.datapath, f"Train_{filename_train_test_dataframe_postfix}.csv"))
    elif data_type == 'CXR':
        filename_train_test_dataframe_postfix = f'CXR'
        save_pickle_filename = f'train_validation_index_pickle_{n_folds}folds' \
                               f'_random_seed_{cfg.seed}.p'
        if not os.path.isfile(os.path.join(cfg.datapath, preprocessed_data_filename)):
            df_meta = create_df_CXR_forensic(cfg, "ID_AGE_PATH.csv", save=True,
                                             preprocessed_filename=preprocessed_data_filename)
        else:
            df_meta = pd.read_csv(os.path.join(cfg.datapath, preprocessed_data_filename))

        if not os.path.isfile(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv")):
            df_train, df_test = split_train_test_acc2list(cfg, df_meta, "train_val_list.txt", "test_list.txt",
                                                          save=True,
                                                          save_filename=filename_train_test_dataframe_postfix)
        else:
            df_test = pd.read_csv(os.path.join(cfg.datapath, f"Test_{filename_train_test_dataframe_postfix}.csv"))
            df_train = pd.read_csv(
                os.path.join(cfg.datapath, f"Train_{filename_train_test_dataframe_postfix}.csv"))
    else:
        raise ValueError(f'Not support this {data_type} data type')

    print(f"Number of test data: {df_test.shape[0]}")
    print(
        f"Number of train data: {df_train.shape[0]}")

    # Training and test indices
    if not os.path.isfile(os.path.join(cfg.datapath, save_pickle_filename)):
        split_cv_train_val(cfg, df_train, save_pickle=True, save_filename=save_pickle_filename)
        index = pickle.load(open(os.path.join(cfg.datapath, save_pickle_filename), "rb"))
    else:
        index = pickle.load(open(os.path.join(cfg.datapath, save_pickle_filename), "rb"))

    do_train_reid_img(cfg, df_train, index)

def do_train_reid_img(cfg, df_meta, index):
    # Configuration
    arch_name = cfg.backbone_model
    i_fold = cfg.i_fold
    device = cfg.device
    n_epochs = cfg.noepochs
    test_site = cfg.test_site
    print(f'Number of epochs: {n_epochs}')
    pid = cfg.personal_id

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

    # print(f'>>> Training transformations: <<<\n{train_transforms.to_yaml()}')
    # print(f'>>> Validation transformations: <<<\n{valid_transforms.to_yaml()}')

    print(saved_model_name)
    train_dict, eval_dict = data_loader_each_fold(cfg, i_fold, df_meta, index, train_transforms,
                                                  valid_transforms, return_dict=True)

    # Architecture
    model = BackboneModel(cfg)

    print(model)
    # Running training and validation
    main_process(cfg, model,
                 train_dict,
                 eval_dict,
                 saved_model_name,
                 device=device,
                 n_epochs=n_epochs)


def main_process(cfg, model, train_dict, eval_dict, model_name, device='cpu', n_epochs=10):
    model.to(device)
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    cfg.vars.best_ap = -1e8
    cfg.vars.best_cmc = -1e8
    if cfg.distance_loss == 'cosine':
        distance = distances.CosineSimilarity()
    elif cfg.distance_loss == 'l2':
        distance = distances.LpDistance()

    train_loader = train_dict['data_loader']
    eval_loader = eval_dict['data_loader']
    eval_df = eval_dict['dataframe']
    query = eval_df[['Path', 'PID', 'Encode_Visit']].values.tolist()

    reducer = reducers.ThresholdReducer(low=0)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    loss_func = {'Triplet': TripletCustomMarginLoss(margin=cfg.margin.m_loss, distance=distance, reducer=reducer),
                 'Triplet_original': losses.TripletMarginLoss(margin=cfg.margin.m_loss, distance=distance,
                                                              reducer=reducer),
                 'CE': CrossEntropyLoss(),
                 'BCE': BCEWithLogitsLoss(),
                 'HG': HingeLoss(), 'Cos': CosineEmbeddingLoss(margin=cfg.margin.m_loss),
                 'originalBCE': BCELoss(),
                 'LowerBoundLoss': LowerBoundLoss(),
                 'LogSumExpLoss': LogSumExpLoss(),
                 'SoftTriplet': losses.SoftTripleLoss(num_classes=len(train_loader) * cfg.batchsize,
                                                      embedding_size=cfg.img_out_features, margin=cfg.margin.m_loss),
                 'ArcFaceLoss': losses.ArcFaceLoss(num_classes=len(train_loader) * cfg.batchsize,
                                                   embedding_size=cfg.img_out_features, margin=cfg.margin.m_loss,
                                                   scale=64),
                 'ContrastiveLoss': losses.ContrastiveLoss(neg_margin=cfg.margin.delta_n,
                                                           pos_margin=cfg.margin.delta_p),
                 'TupletLoss': losses.MultipleLosses(
                     [losses.TupletMarginLoss(margin=cfg.margin.m_loss, distance=distance, reducer=reducer),
                      losses.IntraPairVarianceLoss(distance=distance, reducer=reducer)], weights=[1, 0.5]),
                 'LiftedStructureLoss': losses.LiftedStructureLoss(neg_margin=cfg.margin.delta_n,
                                                                   pos_margin=cfg.margin.delta_p)}
    mining_func = {
        "TripletMargin": TripletMarginMiner(margin=cfg.margin.m_loss, beta_n=cfg.margin.beta, distance=distance,
                                            type_of_triplets=cfg.type_of_triplets),
        "TripletMargin_lib": miners.TripletMarginMiner(margin=cfg.margin.m_loss, distance=distance,
                                                       type_of_triplets=cfg.type_of_triplets),
        "Angular": miners.AngularMiner(angle=cfg.margin.m_loss),
        "PairMargin": miners.PairMarginMiner(pos_margin=cfg.margin.delta_p, neg_margin=cfg.margin.delta_n),
        "TripletMargin_easy": miners.TripletMarginMiner(margin=cfg.margin.m_loss, distance=distance,
                                                        type_of_triplets='easy'),
        "TripletMargin_auto": TripletAutoMarginMiner(distance=distance, margin=cfg.margin.m_loss,
                                                     type_of_triplets=cfg.type_of_triplets,
                                                     k=cfg.k_param_automargin, k_n=cfg.k_n_param_autobeta,
                                                     mode=cfg.automargin_mode),
        "TripletAdaptive": TripletAdaptiveMiner(distance=distance,
                                                type_of_triplets=cfg.type_of_triplets,
                                                k=cfg.k_param_automargin,
                                                mode=cfg.automargin_mode),
        "SCT": TripletSCTMiner(),
        "AutoParams": TripletAutoParamsMiner(distance=distance, margin_init=cfg.margin.m_loss,
                                             beta_init=cfg.margin.beta,
                                             type_of_triplets=cfg.type_of_triplets,
                                             k=cfg.k_param_automargin, k_n=cfg.k_n_param_autobeta,
                                             k_p=cfg.k_p_param_autobeta,
                                             mode=cfg.automargin_mode)
    }
    list_an_dist = {}
    list_ap_dist = {}
    for epoch_id in range(n_epochs):
        if cfg.save_distribution:
            an_dist, ap_dist = train_loop(cfg, model, loss_func, mining_func, optimizer, train_loader, epoch_id, device)
            list_an_dist[f'epoch{epoch_id}'] = an_dist
            list_ap_dist[f'epoch{epoch_id}'] = ap_dist
        train_loop(cfg, model, loss_func, mining_func, optimizer, train_loader, epoch_id, device)
        eval_loop(cfg, model, optimizer, eval_loader, query, epoch_id, device, model_name=model_name)
    if cfg.save_distribution:
        with open(os.path.join(os.getcwd(), f'List_ap_distribution.p'), 'wb') as f:
            pickle.dump(list_ap_dist, f)
        with open(os.path.join(os.getcwd(), f'List_an_distribution.p'), 'wb') as f:
            pickle.dump(list_an_dist, f)


if __name__ == '__main__':
    main()


