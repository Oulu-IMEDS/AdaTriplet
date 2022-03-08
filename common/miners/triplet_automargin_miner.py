import numpy as np
import scipy
import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from common.miners.triplet_margin_miner import TripletMarginMiner
from common.utils import to_cpu


class TripletAutoMarginMiner(TripletMarginMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0, k=2, k_n=2, type_of_triplets="all", mode=False, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.beta_n = 0
        self.type_of_triplets = type_of_triplets
        self.batch_id = None
        self.mode = mode
        self.reset()
        self.mean = 0
        self.std = 0
        if mode == 'exp' or mode == 'linear':
            self.k = 1
        else:
            self.k = k
        self.k_n = k_n

    def reset(self):
        self.ap_an_dists = []
        self.an_dists = []
        self.ap_dists = []

    def get_ap_an_dists(self):
        if len(self.ap_an_dists) > 0:
            ap_an_dists_concat = np.concatenate(self.ap_an_dists, 0).flatten()
            return ap_an_dists_concat

    def get_an_dists(self, mode=None):
        if len(self.an_dists) > 0:
            an_dists_concat = np.concatenate(self.an_dists, 0).flatten()
            return an_dists_concat

    def get_ap_dists(self, mode=None):
        if len(self.ap_dists) > 0:
            ap_dists_concat = np.concatenate(self.ap_dists, 0).flatten()
            return ap_dists_concat

    def update_ap_an(self, dists):
        dists = to_cpu(dists)
        self.ap_an_dists.append(dists)

    def update_an(self, an_dists):
        an_dists = to_cpu(an_dists)
        self.an_dists.append(an_dists)

    def update_ap(self, ap_dists):
        ap_dists = to_cpu(ap_dists)
        self.ap_dists.append(ap_dists)

    def compute_params(self):
        self.compute_margin()
        self.compute_beta_n()
        self.reset()

    def compute_margin(self):
        if len(self.ap_an_dists) > 0:
            self.ap_an_dists = np.concatenate(self.ap_an_dists, 0).flatten()

            # mean = np.mean(self.ap_an_dists)
            # mode = scipy.stats.mode(self.ap_an_dists)[0].item()
            # median = np.quantile(self.ap_an_dists, 0.5)
            # diff = self.ap_an_dists - mean
            # std = np.std(self.ap_an_dists)
            # z_score = diff/std
            # skews = np.mean(np.power(z_score, 3.0))
            # kurtois = np.mean(np.power(z_score, 4.0)) - 3.0

            median = np.quantile(self.ap_an_dists, 0.5)
            mean = np.mean(self.ap_an_dists)
            std = np.std(self.ap_an_dists)
            skews = (3 * (mean - median)) / std
            min = np.min(self.ap_an_dists)
            mode = scipy.stats.mode(self.ap_an_dists)[0].item()
            diff = self.ap_an_dists - mean
            z_score = diff / std
            kurtois = np.mean(np.power(z_score, 4.0)) - 3.0

            if self.mode == 'Q1':
                self.margin = np.quantile(self.ap_an_dists, 0.25)
            elif self.mode == 'Q2':
                self.margin = np.quantile(self.ap_an_dists, 0.5)
            else:
                self.margin = mean / self.k

                # self.margin = self.margin/self.k
            if "adaptive" not in self.mode:
                self.margin = max(0, self.margin)
            self.mean = mean
            self.std = std
            print(f'Margin: {self.margin}')
            print(f'2nd P Skewness: {skews}')
            print(f'Min: {min}')
            print(f'Std: {std}')
            print(f'Mean: {mean}')
            print(f'Median: {median}')
            print(f'Mode: {mode}')
            print(f'Kurtois: {kurtois}')

    def get_margin(self):
        return self.margin

    def compute_beta_n(self):
        if len(self.an_dists) > 0:
            self.an_dists = np.concatenate(self.an_dists, 0).flatten()
            mean = np.mean(self.an_dists)
            self.beta_n = (1 - mean) / self.k_n
            print(f'Beta_n: {self.beta_n}')
            print(f'Mean an dist: {mean}')

    def get_beta_n(self):
        return self.beta_n

    def set_epoch_id_batch_id(self, epoch_id, batch_id):
        self.batch_id = batch_id
        self.epoch_id = epoch_id

    def set_k_value(self):
        if self.mode == 'exp' and self.k < 0:
            self.k = self.k - 0.1
        else:
            self.k = self.k - 0.1
        return self.k

    def mine(self, embeddings, labels, ref_emb, ref_labels):

        if self.batch_id == 0:
            self.compute_params()

        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )
        self.update_ap_an(triplet_margin)
        # self.update_an(an_dist)
        # self.update_ap(ap_dist)

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0
        empty_tuple = (torch.tensor([]), torch.tensor([]))
        return ((
                    anchor_idx[threshold_condition],
                    positive_idx[threshold_condition],
                    negative_idx[threshold_condition],
                ), empty_tuple, empty_tuple)


class TripletAutoParamsMiner(TripletMarginMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin_init, beta_init, k=2, k_n=2, k_p=2, type_of_triplets="all", mode='normal', **kwargs):
        super().__init__(**kwargs)
        self.margin = margin_init
        self.beta_n = beta_init
        self.beta_p = 0
        self.type_of_triplets = type_of_triplets
        self.batch_id = None
        self.mode = mode
        self.reset()
        self.mean = 0
        self.std = 0
        self.num_negative_pairs = 0
        if mode == 'exp' or mode == 'linear':
            self.k = 1
        else:
            self.k = k
        self.k_n = k_n
        self.k_p = k_p

    def reset(self):
        self.ap_an_dists = []
        self.an_dists = []
        self.ap_dists = []
        self.total_an_dists = []
        self.total_ap_dists = []

    def get_ap_an_dists(self):
        if len(self.ap_an_dists) > 0:
            ap_an_dists_concat = np.concatenate(self.ap_an_dists, 0).flatten()
            return ap_an_dists_concat

    def get_an_dists(self, mode='mined'):
        if len(self.an_dists) > 0 or len(self.total_an_dists) > 0:
            if mode == 'mined':
                an_dists_concat = np.concatenate(self.an_dists, 0).flatten()
                return an_dists_concat
            elif mode == 'total':
                total_an_dists_concat = np.concatenate(self.total_an_dists, 0).flatten()
                return total_an_dists_concat

    def get_ap_dists(self, mode='mined'):
        if len(self.ap_dists) > 0 or len(self.total_ap_dists) > 0:
            if mode == 'mined':
                ap_dists_concat = np.concatenate(self.ap_dists, 0).flatten()
                return ap_dists_concat
            elif mode == 'total':
                total_ap_dists_concat = np.concatenate(self.total_ap_dists, 0).flatten()
                return total_ap_dists_concat

    def get_num_negative_pairs(self):
        return self.num_negative_pairs

    def get_num_positive_pairs(self):
        return self.num_positive_pairs

    def update_ap_an(self, dists):
        dists = to_cpu(dists)
        self.ap_an_dists.append(dists)

    def update_an(self, an_dists, mode='mined'):
        if mode == 'mined':
            an_dists = to_cpu(an_dists)
            self.an_dists.append(an_dists)
        elif mode == 'total':
            an_dists = to_cpu(an_dists)
            self.total_an_dists.append(an_dists)

    def update_ap(self, ap_dists, mode='mined'):
        if mode == 'mined':
            ap_dists = to_cpu(ap_dists)
            self.ap_dists.append(ap_dists)
        elif mode == 'total':
            ap_dists = to_cpu(ap_dists)
            self.total_ap_dists.append(ap_dists)

    def compute_params(self):
        self.compute_margin()
        self.compute_beta_n()
        if self.mode == "add-ap":
            self.compute_beta_p()
        self.reset()

    def compute_margin(self):
        if len(self.ap_an_dists) > 0:
            self.ap_an_dists = np.concatenate(self.ap_an_dists, 0).flatten()

            # mean = np.mean(self.ap_an_dists)
            # mode = scipy.stats.mode(self.ap_an_dists)[0].item()
            # median = np.quantile(self.ap_an_dists, 0.5)
            # diff = self.ap_an_dists - mean
            # std = np.std(self.ap_an_dists)
            # z_score = diff/std
            # skews = np.mean(np.power(z_score, 3.0))
            # kurtois = np.mean(np.power(z_score, 4.0)) - 3.0

            median = np.quantile(self.ap_an_dists, 0.5)
            mean = np.mean(self.ap_an_dists)
            std = np.std(self.ap_an_dists)
            skews = (3 * (mean - median)) / std
            min = np.min(self.ap_an_dists)
            mode = scipy.stats.mode(self.ap_an_dists)[0].item()
            diff = self.ap_an_dists - mean
            z_score = diff / std
            kurtois = np.mean(np.power(z_score, 4.0)) - 3.0

            if self.mode == 'Q1':
                self.margin = np.quantile(self.ap_an_dists, 0.25)
            elif self.mode == 'Q2':
                self.margin = np.quantile(self.ap_an_dists, 0.5)
            else:
                self.margin = mean / self.k

            self.margin = max(0, self.margin)
            self.mean = mean
            self.std = std
            print(f'Margin: {self.margin}')
            print(f'2nd P Skewness: {skews}')
            print(f'Min: {min}')
            print(f'Std: {std}')
            print(f'Mean: {mean}')
            print(f'Median: {median}')
            print(f'Mode: {mode}')
            print(f'Kurtois: {kurtois}')

    def get_margin(self):
        return self.margin

    def compute_beta_n(self):
        if len(self.an_dists) > 0:
            self.an_dists = np.concatenate(self.an_dists, 0).flatten()
            mean = np.mean(self.an_dists)
            max = np.max(self.an_dists)

            if self.mode == 'Q1':
                self.beta_n = np.quantile(self.an_dists, 0.75)
            elif self.mode == 'Q2':
                self.beta_n = np.quantile(self.an_dists, 0.5)
            else:
                self.beta_n = 1 - ((1 - mean) / self.k_n)

            print(f'Beta_n: {self.beta_n}')
            print(f'Mean an dist: {mean}')
            print(f'Max an dist: {max}')

    def compute_beta_p(self):
        if len(self.ap_dists) > 0:
            self.ap_dists = np.concatenate(self.ap_dists, 0).flatten()
            mean = np.mean(self.ap_dists)
            min = np.min(self.ap_dists)
            self.beta_p = mean / self.k_p
            print(f'Beta_p: {self.beta_p}')
            print(f'Mean ap dist: {mean}')
            print(f'Min ap dist: {min}')

    def get_beta_n(self):
        return self.beta_n

    def get_beta_p(self):
        return self.beta_p

    def set_epoch_id_batch_id(self, epoch_id, batch_id):
        self.batch_id = batch_id
        self.epoch_id = epoch_id

    def set_k_value(self):
        if self.mode == 'exp' and self.k < 0:
            self.k = self.k - 0.1
        else:
            self.k = self.k - 0.1
        return self.k

    def mine(self, embeddings, labels, ref_emb, ref_labels):

        if "adaptive" not in self.mode:
            if self.batch_id == 0:
                self.compute_params()

        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        self.update_ap_an(triplet_margin)
        self.update_an(an_dist, mode='total')
        self.update_ap(ap_dist, mode='total')

        neg_pairs_idx = torch.stack((anchor_idx, negative_idx)).T
        unique_neg_pairs_idx = list(set([(c, b) if c <= b else (b, c) for c, b in neg_pairs_idx.tolist()]))
        anchor_neg_pairs_idx = torch.tensor([x[0] for x in unique_neg_pairs_idx], dtype=torch.int64)
        neg_pairs_idx = torch.tensor([x[1] for x in unique_neg_pairs_idx], dtype=torch.int64)
        neg_pairs_dist_unique = mat[anchor_neg_pairs_idx, neg_pairs_idx]
        self.update_an(neg_pairs_dist_unique)

        if self.mode == 'add-ap':
            pos_pairs_idx = torch.stack((anchor_idx, positive_idx)).T
            unique_pos_pairs_idx = list(set([(c, b) if c <= b else (b, c) for c, b in pos_pairs_idx.tolist()]))
            anchor_pos_pairs_idx = torch.tensor([x[0] for x in unique_pos_pairs_idx], dtype=torch.int64)
            pos_pairs_idx = torch.tensor([x[1] for x in unique_pos_pairs_idx], dtype=torch.int64)
            pos_pairs_dist_unique = mat[anchor_pos_pairs_idx, pos_pairs_idx]
            self.update_ap(pos_pairs_dist_unique)

        if self.mode == 'adaptive' or self.mode == 'adaptiveNC':
            self.compute_params()

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0
        indices_triplets = (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

        neg_pairs_condition = neg_pairs_dist_unique >= self.beta_n
        self.num_negative_pairs = len(anchor_neg_pairs_idx[neg_pairs_condition])
        indices_negative_pairs = (anchor_neg_pairs_idx[neg_pairs_condition], neg_pairs_idx[neg_pairs_condition])
        # indices_negative_pairs = (anchor_pairs_idx, neg_pairs_idx)

        if self.mode == 'add-ap':
            pos_pairs_condition = pos_pairs_dist_unique <= self.beta_p
            self.num_positive_pairs = len(anchor_pos_pairs_idx[pos_pairs_condition])
            indices_positve_pairs = (anchor_pos_pairs_idx[pos_pairs_condition], pos_pairs_idx[pos_pairs_condition])
        else:
            indices_positve_pairs = (torch.tensor([]), torch.tensor([]))

        indices = (indices_triplets, indices_negative_pairs, indices_positve_pairs)
        return indices


class TripletSCTMiner(TripletMarginMiner):

    def __init__(self, type_of_triplets="all", scheduler=False, **kwargs):
        super().__init__(**kwargs)
        self.type_of_triplets = type_of_triplets
        self.batch_id = None
        self.scheduler = scheduler
        self.mean = 0
        self.std = 0

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        condition = an_dist > ap_dist
        negative_pair_idx = (anchor_idx[condition],
                             negative_idx[condition],)
        self.num_negative_pairs = len(anchor_idx[condition])
        triplet_idx = (anchor_idx[~condition],
                       positive_idx[~condition],
                       negative_idx[~condition],)
        self.num_triplets = len(anchor_idx[~condition])

        return (triplet_idx, negative_pair_idx)


class TripletAdaptiveMiner(TripletMarginMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, k=2, type_of_triplets="all", mode=False, **kwargs):
        super().__init__(**kwargs)
        self.margin = 0
        self.type_of_triplets = type_of_triplets
        self.batch_id = None
        self.mode = mode
        self.reset()
        self.mean = 0
        self.std = 0
        self.k = k

    def reset(self):
        self.ap_an_dists = []
        self.an_dists = []
        self.ap_dists = []

    def get_ap_an_dists(self):
        if len(self.ap_an_dists) > 0:
            ap_an_dists_concat = np.concatenate(self.ap_an_dists, 0).flatten()
            return ap_an_dists_concat

    def get_an_dists(self, mode=None):
        if len(self.an_dists) > 0:
            an_dists_concat = np.concatenate(self.an_dists, 0).flatten()
            return an_dists_concat

    def get_ap_dists(self, mode=None):
        if len(self.ap_dists) > 0:
            ap_dists_concat = np.concatenate(self.ap_dists, 0).flatten()
            return ap_dists_concat

    def update_ap_an(self, dists):
        dists = to_cpu(dists)
        self.ap_an_dists.append(dists)


    def compute_margin(self):
        if len(self.ap_an_dists) > 0:
            self.ap_an_dists = np.concatenate(self.ap_an_dists, 0).flatten()

            median = np.quantile(self.ap_an_dists, 0.5)
            mean = np.mean(self.ap_an_dists)
            std = np.std(self.ap_an_dists)
            skews = (3 * (mean - median)) / std
            min = np.min(self.ap_an_dists)
            mode = scipy.stats.mode(self.ap_an_dists)[0].item()
            diff = self.ap_an_dists - mean
            z_score = diff / std
            kurtois = np.mean(np.power(z_score, 4.0)) - 3.0

            if self.mode == 'weakly':
                self.margin = (2 - 2 * self.ap_an_dists) / (4 - self.k) + self.k
            else:
                self.margin = mean / self.k

                # self.margin = self.margin/self.k
            if self.mode != 'weakly':
                self.margin = max(0, self.margin)
            self.mean = mean
            self.reset()

    def get_margin(self):
        return self.margin

    def compute_beta_n(self):
        if len(self.an_dists) > 0:
            self.an_dists = np.concatenate(self.an_dists, 0).flatten()
            mean = np.mean(self.an_dists)
            self.beta_n = (1 - mean) / self.k_n
            print(f'Beta_n: {self.beta_n}')
            print(f'Mean an dist: {mean}')

    def get_beta_n(self):
        return self.beta_n

    def set_epoch_id_batch_id(self, epoch_id, batch_id):
        self.batch_id = batch_id
        self.epoch_id = epoch_id

    def set_k_value(self):
        if self.scheduler == 'exp' and self.k < 0:
            self.k = self.k - 0.1
        else:
            self.k = self.k - 0.1
        return self.k

    def set_mode(self, mode):
        self.mode = mode

    def mine(self, embeddings, labels, ref_emb, ref_labels):

        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )
        self.update_ap_an(triplet_margin)
        self.compute_margin()

        if self.mode == 'weakly':
            self.margin = torch.tensor(self.margin, device=embeddings.device)
            if self.type_of_triplets == "easy":
                threshold_condition = triplet_margin - self.margin > 0
            else:
                threshold_condition = triplet_margin - self.margin <= 0
            self.margin = self.margin[threshold_condition]

        else:
            if self.type_of_triplets == "easy":
                threshold_condition = triplet_margin > self.margin
            else:
                threshold_condition = triplet_margin <= self.margin
                if self.type_of_triplets == "hard":
                    threshold_condition &= triplet_margin <= 0
                elif self.type_of_triplets == "semihard":
                    threshold_condition &= triplet_margin > 0

        empty_tuple = (torch.tensor([]), torch.tensor([]))
        return ((
                    anchor_idx[threshold_condition],
                    positive_idx[threshold_condition],
                    negative_idx[threshold_condition],
                ), empty_tuple, empty_tuple)
