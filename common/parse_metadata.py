import glob
import os

import cv2
import pandas as pd
from sas7bdat import SAS7BDAT
from tqdm import tqdm


def create_df_patient(cfg, raw_folder, save=False):
    parse_file_baseline = pd.read_csv(os.path.join(cfg.metadatapath, "AllClinical00.csv"), sep='|', header=0)
    df_ID_AGE = parse_file_baseline.filter(items=['ID', 'V00AGE'])
    part_site_file = read_sas7bdata_pd(os.path.join(cfg.metadatapath, 'enrollees.sas7bdat'))
    df_ID_SITE = part_site_file.filter(items=['ID', 'V00SITE']).rename(index=str, columns={
        'V00SITE': 'Site'})
    df_ID_SITE.ID = df_ID_SITE.ID.astype(int)
    # df_ID_SITE.SIDE = df_ID_SITE.SITE.astype(str)

    img_path = glob.glob(os.path.join(cfg.image_raw_path, raw_folder + '/*/*/*/*/*/*'))
    data_OAI = []
    for img in tqdm(img_path, total=len(img_path)):
        row_dict = {}
        patient = int(img.split('/')[-4])
        visit = img.split('/')[-6].split('m')[0]
        folder_group = img.split('/')[-5]
        path = img.split(cfg.image_raw_path)[1]
        row_dict['ID'] = patient
        row_dict['Visit'] = visit
        row_dict['Folder'] = folder_group
        row_dict['Path'] = path
        age_baseline = int(df_ID_AGE[df_ID_AGE['ID'] == patient]['V00AGE'].to_numpy())
        if visit == '00':
            age = age_baseline
        elif visit == '12':
            age = age_baseline + 1
        elif visit == '24':
            age = age_baseline + 2
        elif visit == '36':
            age = age_baseline + 3
        elif visit == '48':
            age = age_baseline + 4
        elif visit == '72':
            age = age_baseline + 6
        elif visit == '96':
            age = age_baseline + 8
        row_dict['Age'] = age
        data_OAI.append(row_dict)

    df_OAI_raw = pd.DataFrame(data=data_OAI)
    df_OAI_patient = pd.merge(df_OAI_raw, df_ID_SITE, on="ID", how='left')
    df_OAI_patient.drop_duplicates(subset=['Path'], inplace=True)
    if save == True:
        save_dataframe_csv(df_OAI_patient, 'ID_AGE_SITE_PATH.csv', cfg.datapath)
    return df_OAI_patient


def create_df_clinical(cfg, save=False):
    encode_visid = {'00': 0, '01': 12, '03': 24, '05': 36, '06': 48, '08': 72, '10': 96}
    visit_id = ['00', '01', '03', '05', '06', '08', '10']
    clinial_df_final = pd.DataFrame()
    data_enrollees = read_sas7bdata_pd(os.path.join(cfg.metadatapath, 'enrollees.sas7bdat'))
    for index, id in enumerate(visit_id):
        data_clinical = read_sas7bdata_pd(os.path.join(cfg.metadatapath, f'allclinical{id}.sas7bdat'))

        clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

        # Visit, Site
        clinical_data_oai['Visit'] = encode_visid[id]
        clinical_data_oai['Site'] = clinical_data_oai['V00SITE']

        # Sex
        clinical_data_oai.ID = clinical_data_oai.ID.values.astype(int)
        clinical_data_oai['SEX'] = 2 - clinical_data_oai['P02SEX']

        if id == '00':
            AGE_col = 'V00AGE'
            BMI_col = 'P01BMI'
            HEIGHT_col = 'P01HEIGHT'
            WEIGHT_col = 'P01WEIGHT'
            INJL_col = 'P01INJL'
            INJR_col = 'P01INJR'
            SURGL_col = 'P01KSURGL'
            SURGR_col = 'P01KSURGR'
            WOMACL_col = 'V00WOMTSL'
            WOMACR_col = 'V00WOMTSR'
        else:
            AGE_col = f'V{id}AGE'
            BMI_col = f'V{id}BMI'
            HEIGHT_col = f'V{id}HEIGHT'
            WEIGHT_col = f'V{id}WEIGHT'
            INJL_col = f'V{id}INJL12'
            INJR_col = f'V{id}INJR12'
            SURGL_col = f'V{id}KSRGL12'
            SURGR_col = f'V{id}KSRGR12'
            WOMACL_col = f'V{id}WOMTSL'
            WOMACR_col = f'V{id}WOMTSR'
        # Age, BMI
        clinical_data_oai['AGE'] = clinical_data_oai[f'{AGE_col}']
        clinical_data_oai['BMI'] = clinical_data_oai[f'{BMI_col}']
        if f'{HEIGHT_col}' in clinical_data_oai.columns:
            clinical_data_oai['HEIGHT'] = clinical_data_oai[f'{HEIGHT_col}']
        else:
            df_height = clinial_df_final[
                (clinial_df_final['Visit'] == encode_visid[visit_id[index - 1]]) & (clinial_df_final['SIDE'] == 'R')][
                ['ID', 'HEIGHT']]
            df_height['ID'] = df_height['ID'].astype(int)
            clinical_data_oai = pd.merge(clinical_data_oai, df_height, on="ID", how='left')
        clinical_data_oai['WEIGHT'] = clinical_data_oai[f'{WEIGHT_col}']

        clinical_data_oai_left = clinical_data_oai.copy()
        clinical_data_oai_right = clinical_data_oai.copy()

        # Making side-wise metadata
        clinical_data_oai_left['Side'] = 'L'
        clinical_data_oai_right['Side'] = 'R'

        # Injury (ever had)
        clinical_data_oai_left['INJ'] = clinical_data_oai_left[f'{INJL_col}']
        clinical_data_oai_right['INJ'] = clinical_data_oai_right[f'{INJR_col}']

        # Surgery (ever had)
        clinical_data_oai_left['SURG'] = clinical_data_oai_left[f'{SURGL_col}']
        clinical_data_oai_right['SURG'] = clinical_data_oai_right[f'{SURGR_col}']

        # Total WOMAC score
        clinical_data_oai_left['WOMAC'] = clinical_data_oai_left[f'{WOMACL_col}']
        clinical_data_oai_right['WOMAC'] = clinical_data_oai_right[f'{WOMACR_col}']

        clinical_data_oai = pd.concat((clinical_data_oai_left, clinical_data_oai_right))
        clinical_data_oai.ID = clinical_data_oai.ID.values.astype(int)
        clinial_df_site = clinical_data_oai[
            ['ID', 'Visit', 'Side', 'Site', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'HEIGHT', 'WEIGHT']]
        clinial_df_site.dropna(subset=['AGE'], inplace=True)
        clinial_df_site = clinial_df_site.rename(columns={'Side': 'SIDE'})
        clinial_df_final = pd.concat([clinial_df_final, clinial_df_site])
    if save:
        save_dataframe_csv(clinial_df_final, 'Clinical_data_follow_ups.csv', cfg.datapath)
    return clinial_df_final


def create_df_ROI(cfg, ROI_folder, save=False):
    img_ROI_path = glob.glob(os.path.join(cfg.image_crop_path, ROI_folder + '/*'))
    i = 0
    data_ROI = []
    for img_ROI in tqdm(img_ROI_path, total=len(img_ROI_path)):
        id = img_ROI.split(os.sep)[-1].split('_')[0]
        visit = img_ROI.split(os.sep)[-1].split('_')[1]
        position = img_ROI.split(os.sep)[-1].split('_')[-1].split('.')[0]
        path = img_ROI.split(cfg.image_crop_path)[1]
        row_dict_ROI = {}
        row_dict_ROI['ID'] = int(id)
        row_dict_ROI['Visit'] = int(visit)
        row_dict_ROI['SIDE'] = position
        row_dict_ROI['Path'] = path

        data_ROI.append(row_dict_ROI)
    df_ROI = pd.DataFrame(data=data_ROI)
    if save:
        save_dataframe_csv(df_ROI, cfg.cropset, cfg.datapath)
    return df_ROI


def create_df_kl(cfg, save=False):
    kl_files = glob.glob(os.path.join(cfg.metadatapath, 'kxr_sq_bu*.txt'))
    encode_visid = {'00': 0, '01': 12, '03': 24, '05': 36, '06': 48, '08': 72, '10': 96}
    df_ID_KL = pd.DataFrame(columns=['ID', 'Visit', 'SIDE', 'KL'])
    for file in tqdm(kl_files, total=len(kl_files)):
        print(file)
        parse_file_kl_baseline = pd.read_csv(os.path.join(cfg.metadatapath, file), sep='|', header=0)
        visit_id = file.split('bu')[1].split('.')[0]
        print(visit_id)
        if f'V{visit_id}XRKL' in parse_file_kl_baseline.columns:
            KL_column = f'V{visit_id}XRKL'
        elif f'v{visit_id}XRKL' in parse_file_kl_baseline.columns:
            KL_column = f'v{visit_id}XRKL'
        df_ID_KL_visit = parse_file_kl_baseline.filter(items=['ID', 'SIDE', KL_column]).rename(index=str, columns={
            KL_column: 'KL'})
        df_ID_KL_visit['Visit'] = encode_visid[visit_id]

        df_ID_KL = pd.concat([df_ID_KL, df_ID_KL_visit])

    df_ID_KL['SIDE'] = df_ID_KL['SIDE'].map({1: 'R', 2: 'L'})
    df_ID_KL.drop_duplicates(inplace=True)
    if save:
        save_dataframe_csv(df_ID_KL, cfg.metadata_KLset, cfg.datapath)
    return df_ID_KL


def resize_img_and_create_df_chest(cfg, raw_folder, resized_folder, resized_size, save=False):
    df_meta_data = pd.read_csv(os.path.join(cfg.metadatapath, "Data_Entry_2017_v2020.csv"), header=0)
    imgs_path = glob.glob(os.path.join(cfg.image_raw_path, raw_folder + '/*'))
    resized_path = os.path.join(cfg.image_crop_path, resized_folder)
    data_chest = []
    for img_path in tqdm(imgs_path, total=len(imgs_path)):
        row_dict = {}
        img_idx = img_path.split(f'{raw_folder}{os.sep}')[1]
        if not os.path.exists(os.path.join(resized_path, img_idx)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, resized_size)
            cv2.imwrite(os.path.join(resized_path, img_idx), resized_img)
        patient = int(img_idx.split('_')[0])
        path = os.path.join(resized_folder, img_idx)
        row_dict['Image Index'] = img_idx
        row_dict['Patient ID'] = patient
        row_dict['Path'] = path
        data_chest.append(row_dict)

    df_chest_img = pd.DataFrame(data=data_chest)
    df_patient = pd.merge(df_chest_img, df_meta_data, on=['Patient ID', 'Image Index'], how='left')
    df_patient.drop_duplicates(subset=['Path'], inplace=True)
    df_patient.rename(
        columns={"Patient ID": "ID", "Patient Age": "AGE", "Follow-up #": "Follow-up", "View Position": "SIDE"},
        inplace=True)
    if save == True:
        save_dataframe_csv(df_patient, cfg.cropset, cfg.datapath)
    return df_patient


def save_dataframe_csv(df, folder_name, save_path):
    df.to_csv(os.path.join(save_path, folder_name), index=None)


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])
