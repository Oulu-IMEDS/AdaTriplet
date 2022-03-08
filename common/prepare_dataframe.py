import os

import hydra
import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT
from tqdm import tqdm

from common.parse_metadata import create_df_patient, create_df_kl, create_df_ROI, create_df_clinical, \
    resize_img_and_create_df_chest


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def create_mergedRawCrop_dataframe(cfg, cutoff_age_min=0, cutoff_age_max=120, save=False):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.cropset)):
        df_meta_crop = create_df_ROI(cfg, "ResizedImages", save=True)
    else:
        df_meta_crop = pd.read_csv(os.path.join(cfg.datapath, cfg.cropset))
    if not os.path.exists(os.path.join(cfg.datapath, cfg.rawset)):
        df_meta_raw = create_df_patient(cfg)
    else:
        df_meta_raw = pd.read_csv(os.path.join(cfg.datapath, cfg.rawset))
    df_meta = pd.merge(df_meta_crop, df_meta_raw[['ID', 'Visit', 'Age', 'Site']], on=["ID", "Visit"])
    df_meta.drop_duplicates(subset=['Path'], inplace=True)
    df_meta = df_meta[(df_meta['Age'] >= cutoff_age_min) & (df_meta['Age'] <= cutoff_age_max)]
    df_meta.reset_index(drop=True, inplace=True)
    if save:
        df_meta.to_csv(os.path.join(cfg.datapath, cfg.merged_cropraw_file), index=None)
    return df_meta


def merge_clinical_crop_image(cfg, cutoff_age_min=0, cutoff_age_max=120, save=False):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.cropset)):
        df_meta_crop = create_df_ROI(cfg, "ResizedImages", save=True)
    else:
        df_meta_crop = pd.read_csv(os.path.join(cfg.datapath, cfg.cropset))
    if not os.path.exists(os.path.join(cfg.datapath, cfg.clinicalset)):
        df_meta_clinical = create_df_clinical(cfg, save=True)
    else:
        df_meta_clinical = pd.read_csv(os.path.join(cfg.datapath, cfg.clinicalset))
    df_meta = pd.merge(df_meta_crop, df_meta_clinical, on=['ID', 'Visit', 'SIDE'], how='right')
    df_meta.dropna(subset=['Path'], inplace=True)
    df_meta.drop_duplicates(subset=['Path'], inplace=True)
    df_meta = df_meta[(df_meta['AGE'] >= cutoff_age_min) & (df_meta['AGE'] <= cutoff_age_max)]
    df_meta.reset_index(drop=True, inplace=True)
    if save:
        df_meta.to_csv(os.path.join(cfg.datapath, cfg.merged_crop_clinical_file), index=None)
    return df_meta


def create_group_collumn_acc2age(df, age_space=10):
    # classified age
    df["Group"] = np.nan
    base = age_space
    min_age = df["AGE"].min()
    max_age = df["AGE"].max()
    started_point = min_age
    end_point = max_age
    age_range = np.arange(started_point - 1, end_point, age_space)
    for row_id, row in tqdm(df.iterrows(), total=len(df)):
        age = int(row['AGE'])
        group = np.searchsorted(age_range, age, side='left') - 1
        df.loc[row_id, "Group"] = group
    df.Group = df.Group.astype(int)
    df.drop_duplicates(subset=['Path'], inplace=True)
    return df


def create_classification_dataframe(cfg, age_space=5, applied_kl=False, init_group=True, save=False, filename=None):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.merged_cropraw_file)):
        df_meta = create_mergedRawCrop_dataframe(cfg, save=True)
    else:
        df_meta = pd.read_csv(os.path.join(cfg.datapath, cfg.merged_cropraw_file))
    if init_group:
        df_meta = create_group_collumn_acc2age(df_meta)
    # classified age
    df_meta["Target"] = np.nan
    base = age_space
    min_age = df_meta["Age"].min()
    max_age = df_meta["Age"].max()
    # started_point = min_age//base * base
    # end_point = (max_age//base + 1 )* base
    # age_range = np.arange(started_point - 1, end_point + 1, age_space)
    started_point = min_age
    end_point = max_age
    age_range = np.arange(started_point - 1, end_point, age_space)
    for row_id, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        age = int(row['Age'])
        group = np.searchsorted(age_range, age, side='left') - 1
        df_meta.loc[row_id, "Target"] = group
    # select 1 visit
    if applied_kl:
        if not os.path.exists(os.path.join(cfg.datapath, cfg.metadata_KLset)):
            df_KL = create_df_kl(cfg, save=True)
        else:
            df_KL = pd.read_csv(os.path.join(cfg.datapath, cfg.metadata_KLset))
        # df_meta = pd.merge(df_meta, df_KL[["ID", "Visit", "Position", 'KL']], on=["ID", "Visit", "Position"])
        df_meta = pd.merge(df_KL, df_meta, on=["ID", "Visit", "SIDE"])
    df_meta['Path'] = df_meta['Path'].map(lambda x: x.lstrip('/'))
    df_meta.drop_duplicates(subset=['Path'], inplace=True)
    # df_meta.to_csv(os.path.join(cfg.datapath, 'PreprocessedDataClassification_withKL.csv'), index=None)
    if save:
        if filename is None:
            raise AttributeError("Need to fill filename when saving")
        else:
            df_meta.to_csv(os.path.join(cfg.datapath, filename), index=None)
    return df_meta


def create_regression_dataframe(cfg, applied_kl=False, init_group=True, save=False, filename=None):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.merged_cropraw_file)):
        df_meta = create_mergedRawCrop_dataframe(cfg, cutoff_age_max=80, save=True)
    else:
        df_meta = pd.read_csv(os.path.join(cfg.datapath, cfg.merged_cropraw_file))
    if init_group:
        df_meta = create_group_collumn_acc2age(df_meta)
    df_meta = df_meta.rename(columns={'Age': 'Target'})
    if applied_kl:
        if not os.path.exists(os.path.join(cfg.datapath, 'ID_SIDE_KL.csv')):
            df_KL = create_df_kl(cfg, save=True)
        else:
            df_KL = pd.read_csv(os.path.join(cfg.datapath, 'ID_SIDE_KL.csv'))
        df_meta = pd.merge(df_meta, df_KL[["ID", "Visit", "SIDE", 'KL']], on=["ID", "Visit", "SIDE"], how='left')
    df_meta.drop_duplicates(subset=['Path'], inplace=True)
    df_meta['Path'] = df_meta['Path'].map(lambda x: x.lstrip(os.sep))
    # df_meta.to_csv(os.path.join(cfg.datapath, 'PreprocessedDataClassification_withKL.csv'), index=None)
    if save:
        if filename is None:
            raise AttributeError("Need to fill filename when saving")
        else:
            df_meta.to_csv(os.path.join(cfg.datapath, filename), index=None)
    return df_meta


def create_df_OAI_forensic(cfg, applied_kl=False, init_group=True, save=False, filename=None):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.merged_crop_clinical_file)):
        df_meta = merge_clinical_crop_image(cfg, cutoff_age_max=120, save=True)
    else:
        df_meta = pd.read_csv(os.path.join(cfg.datapath, cfg.merged_crop_clinical_file))
    if init_group:
        df_meta = create_group_collumn_acc2age(df_meta)
    if applied_kl:
        if not os.path.exists(os.path.join(cfg.datapath, cfg.metadata_KLset)):
            df_KL = create_df_kl(cfg, save=True)
        else:
            df_KL = pd.read_csv(os.path.join(cfg.datapath, cfg.metadata_KLset))
        df_meta = pd.merge(df_meta, df_KL[["ID", "Visit", "SIDE", 'KL']], on=["ID", "Visit", "SIDE"], how='left')
    df_meta.drop_duplicates(subset=['Path'], inplace=True)
    df_meta['Path'] = df_meta['Path'].map(lambda x: x.lstrip(os.sep))
    # df_meta.to_csv(os.path.join(cfg.datapath, 'PreprocessedDataClassification_withKL.csv'), index=None)
    if save:
        if filename is None:
            raise AttributeError("Need to fill filename when saving")
        else:
            df_meta.to_csv(os.path.join(cfg.datapath, filename), index=None)
    return df_meta


def merge_results_to_preprocess_data(cfg, df_preprocess, save=True, filename=None):
    model = cfg.run_model
    KL_grade_train = cfg.KL.KL_grade_train
    KL_grade_test = cfg.KL.KL_grade_test
    cutoff_age_min = cfg.cutoff_age_min
    cutoff_age_max = cfg.cutoff_age_max

    min_age = df_preprocess["Target"].min()
    df_merge_pred_KL = pd.DataFrame()
    for site in ["A", "B", "C", "D", "E"]:
        predicted_filename_sites = f'{model}_predicted_results_site{site}_train-KL{KL_grade_train}_test-KL{KL_grade_test}_age{cutoff_age_min}{cutoff_age_max}.csv'
        df_pred = pd.read_csv(os.path.join(cfg.result_path, predicted_filename_sites))
        df_pred["Pred"] = df_pred["Pred"] + min_age
        df_preprocess_site = df_preprocess[df_preprocess["Site"] == site]
        df_merge_pred_KL_site = pd.merge(df_preprocess_site, df_pred[["Path", "Pred", "Site"]], on=["Path", "Site"],
                                         how='left')
        df_merge_pred_KL_site.drop_duplicates(subset=['Path'], inplace=True)
        df_merge_pred_KL = pd.concat([df_merge_pred_KL, df_merge_pred_KL_site], axis=0)

    df_merge_pred_KL.rename(columns={'Target': 'Age'}, inplace=True)
    df_merge_pred_KL.dropna(subset=["KL"], inplace=True)
    if save:
        df_merge_pred_KL.to_csv(os.path.join(cfg.result_path, filename), index=None)
    return df_merge_pred_KL


def create_df_pairs(cfg, meta_data, n_pairs, n_pos_pairs, n_neg_pairs, save=False, filename=None):
    pairs_df = pd.DataFrame()
    for index, entry in tqdm(meta_data.iterrows(), total=len(meta_data)):
        data_same_ID = meta_data[(meta_data['ID'] == entry['ID']) & (meta_data['SIDE'] == entry['SIDE']) & (
                meta_data['Visit'] != entry['Visit'])]
        if len(data_same_ID) < n_pos_pairs:
            pos_pairs = data_same_ID
        else:
            pos_pairs = data_same_ID.sample(n=n_pos_pairs)
        data_diff_ID = meta_data[meta_data['ID'] != entry['ID']]
        if len(data_diff_ID) < n_neg_pairs:
            neg_pairs = data_diff_ID
        else:
            neg_pairs = data_diff_ID.sample(n=n_neg_pairs)

        count_neg_pairs = len(neg_pairs)
        count_pos_pairs = len(pos_pairs)
        count_neg_pos_pairs = 2 * count_pos_pairs + count_neg_pairs
        aug_pairs = pd.DataFrame(columns=['ID', 'Visit', 'SIDE', 'Path'])
        for i in range(n_pairs - count_neg_pos_pairs):
            aug_pairs.loc[i] = entry
        pairs_entry = pd.concat([pos_pairs, neg_pairs, aug_pairs])
        pairs_entry = pairs_entry.sample(frac=1).reset_index(drop=True)
        pairs_entry['ID_anchor'] = entry['ID']
        pairs_entry['Visit_anchor'] = entry['Visit']
        pairs_entry['SIDE_anchor'] = entry['SIDE']
        pairs_df = pd.concat([pairs_df, pairs_entry])
    if save:
        pairs_df.to_csv(os.path.join(cfg.datapath, filename), index=None)
    return pairs_df


def mark_visit_by_age(row, ref_df):
    visit0_age = ref_df[ref_df['ID'] == row['ID']]['AGE'].item()
    diff_age = row['AGE'] - visit0_age
    return diff_age


def create_df_CXR_forensic(cfg, init_group=True, save=False, preprocessed_filename=None):
    if not os.path.exists(os.path.join(cfg.datapath, cfg.cropset)):
        df = resize_img_and_create_df_chest(cfg, "ChestImages", "ResizedCXRImages", (300, 300), save=True)
    else:
        df = pd.read_csv(os.path.join(cfg.datapath, cfg.cropset))
    df_followup0 = df[df['Follow-up'] == 0][['ID', 'AGE']]
    df['Visit_id'] = df.apply(lambda row: mark_visit_by_age(row, df_followup0), axis=1)
    df['Visit'] = df['Visit_id'] * 12

    if init_group:
        df = create_group_collumn_acc2age(df)

    if save:
        df.to_csv(os.path.join(cfg.datapath, preprocessed_filename), index=None)
    return df


@hydra.main(config_path=os.pardir, config_name="config.yml")
def main(cfg):
    # df = create_df_patient(cfg,'OAI_raw',save=True)
    # print(df.head(10))
    # df_meta = pd.read_csv(os.path.join(cfg.datapath, 'PreprocessedData_Forensic.csv'))
    # df = create_df_pairs(cfg, df_meta, n_pairs=10, n_pos_pairs=2,n_neg_pairs=4, save=True, filename='Forensic_pairs.csv')
    # print(df.head(10))

    df = create_df_CXR_forensic(cfg, "ID_AGE_PATH.csv", save=True)
    print(df)

    # resize_img(cfg, 'ID_PATH_AGE_VISIT.csv', (300,300))

    # df = create_df_clinical(cfg, save=True)
    # print(df.head(10))


if __name__ == '__main__':
    main()
