from __future__ import print_function, absolute_import

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from torch.nn import TripletMarginLoss
import torch.nn.functional as F
from common.metric_learning import factory
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from .evaluation_metrics import cmc, mean_ap
from .evaluation_metrics.ranking import list_out_distance, list_out_rank, mean_ap_at_R


def extract_features(model, data_loader, epoch_id, device, print_freq=1, metric=None):
    model.eval()

    features = OrderedDict()
    labels = OrderedDict()

    n_batches = len(data_loader)
    progress_bar = tqdm(data_loader, total=n_batches, desc=f"Epoch [{epoch_id}][Eval]:")
    for i, batch in enumerate(progress_bar):
        data = batch['data'].to(device)
        # imgs = data[:, 0, 0, :, :, :]
        imgs = data
        fnames = batch['Path']
        pids = batch['pid']

        # outputs = model.forward_img(imgs)
        outputs = model(imgs)
        outputs = outputs.data.cpu()

        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

    return features, labels


def extract_features_query_gallery(model,
                           data_loader, epoch_id,
                           device, return_features_gallery=False):
    model.eval()
    metrics_collector = {'loss': [], 'probs': [], 'preds': [], 'labels': [], 'L1_norm': None}
    features_query = OrderedDict()
    labels_query = OrderedDict()
    features_gallery = OrderedDict()
    labels_gallery = OrderedDict()

    n_batches = len(data_loader)
    progress_bar = tqdm(data_loader, total=n_batches, desc=f"Epoch [{epoch_id}][Eval]:")
    for i, batch in enumerate(progress_bar):
        inputs = batch['data'].to(device)
        bio_features = batch['bio_features']
        bio_profile = batch['bio_data']
        # for var in bio_features:
        #     bio_features[var] = bio_features[var].to(device)
        for var in bio_profile:
            bio_profile[var] = bio_profile[var].to(device)

        imgs = inputs[:, 0, 0, :, :, :]
        fnames = batch['Path']
        pids = batch['pid']

        #delete this part after check
        anchor = inputs[:, 0, 0, :, :, :]
        positive = inputs[:, 0, 1, :, :, :]
        negative = inputs[:, 1, 1, :, :, :]
        bio_features_pos = bio_features['positive'][0]
        bio_features_neg = bio_features['negative'][0]
        final_anchor_features = model(img=anchor, bio_profile=bio_profile, encode_BP=True)
        final_pos_features = model(img=positive, bio_features=bio_features_pos)
        final_neg_features = model(img=negative, bio_features=bio_features_neg)
        loss_func = TripletMarginLoss()
        loss = loss_func(final_anchor_features, final_pos_features, final_neg_features)
        metrics_collector['loss'].append(loss.item())
        metrics_display = {'loss': loss.item(), 'mean_loss': np.mean(metrics_collector['loss'])}
        progress_bar.set_postfix(metrics_display)

        # Forward through the model
        final_features_query_new = model(img=imgs, bio_features=bio_features['anchor'][0])

        for fname, output, pid in zip(fnames, final_features_query_new, pids):
            features_query[fname] = output
            labels_query[fname] = pid

        if return_features_gallery:
            final_features_gallery = model(img=imgs, bio_profile=bio_profile, encode_BP=True)
            for fname, output, pid in zip(fnames, final_features_gallery, pids):
                features_gallery[fname] = output
                labels_gallery[fname] = pid

    if return_features_gallery:
        return features_query, features_gallery
    else:
        return features_query


def pairwise_distance(features, query=None, gallery=None, metric=None, return_features=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    features_query = x
    features_gallery = y
    if metric is not None and metric.algorithm in factory:
        x = metric.transform(x)
        y = metric.transform(y)
    if metric.algorithm in factory :
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
    elif metric.algorithm == 'cosine':
        x_norm = F.normalize(x)
        y_norm = F.normalize(y)
        dist = torch.matmul(x_norm, y_norm.T)
    if return_features:
        return dist, (features_query,features_gallery)
    else:
        return dist

def pairwise_distance_combined_BP(features_query, features_gallery, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features_query)
        x = torch.cat(list(features_query.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features_query[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features_gallery[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), mAP_topk=None, return_cmc_topk=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_scores = cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100)

    list_cmc_topk = []
    print('CMC Scores:')
    for k in cmc_topk:
        if cmc_topk[0] == 1:
            print('  top-{:<4}{:12.1%}'
                  .format(k, cmc_scores[k - 1]))
            list_cmc_topk.append(cmc_scores[k - 1])
        else:
            print(f'  top-{k+1}: {cmc_scores[k]}')
            list_cmc_topk.append(cmc_scores[k])

    # Use the allshots cmc top-1 score for validation criterion
    if return_cmc_topk:
        return mAP, list_cmc_topk
    else:
        return mAP, cmc_scores[0]

def evaluate_mAP_at_R(features_query, features_gallery, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), mAP_topk=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    ml_accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",), k='max_bin_count')

    query_labels = torch.tensor(list(map(int, query_ids)))
    gallery_labels = torch.tensor(list(map(int, gallery_ids)))
    # Compute mean AP
    mAP_R = ml_accuracy_calculator.get_accuracy(query=features_query, reference=features_gallery,
                                                          query_labels=query_labels, reference_labels=gallery_labels,
                                                          embeddings_come_from_same_source=False)

    mAP_R = mAP_R['mean_average_precision_at_r']
    print(f'Mean AP at R: {mAP_R}')

    return mAP_R

def get_distribution(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    an, ap = list_out_distance(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

    return an, ap

def get_rank(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    ranks, indices = list_out_rank(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    for i in range(len(query_ids)):
        query[i].append(ranks[i])
        query[i].append(indices[i,:])


    return query


class Evaluator(object):
    def __init__(self, model, epoch_id, device,
                 combined_BP=False, ):
        super(Evaluator, self).__init__()
        self.model = model
        self.epoch_id = epoch_id
        self.device = device
        self.combined_BP = combined_BP



    def evaluate(self, data_loader, query, gallery, metric=None, return_distmat_features=False):
        if self.combined_BP:
            features_query, features_gallery = extract_features_query_gallery(self.model, data_loader, epoch_id=self.epoch_id,
                                                 device=self.device, return_features_gallery=True )
            distmat = pairwise_distance_combined_BP(features_query, features_gallery, query=query, gallery=gallery, metric=metric)
        else:
            features, _ = extract_features(self.model, data_loader, epoch_id=self.epoch_id, device=self.device)
            distmat, features_query_gallery = pairwise_distance(features, query, gallery, metric=metric, return_features=True)
        if return_distmat_features:
            return distmat, features_query_gallery
        else:
            mAP, cmc_0 = evaluate_all(distmat, query=query, gallery=gallery)
            return mAP, cmc_0
