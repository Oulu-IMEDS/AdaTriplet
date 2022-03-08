import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from common.miners.base_miner import BaseTupleMiner


class TripletMarginMiner(BaseTupleMiner):
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

    def __init__(self, margin=0.2, beta_n=0, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.beta_n = beta_n
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

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
        neg_pairs_idx = torch.stack((anchor_idx, negative_idx)).T
        unique_neg_pairs_idx = list(set([(c, b) if c <= b else (b, c) for c, b in neg_pairs_idx.tolist()]))
        anchor_pairs_idx = torch.tensor([x[0] for x in unique_neg_pairs_idx], dtype=torch.int64)
        neg_pairs_idx = torch.tensor([x[1] for x in unique_neg_pairs_idx], dtype=torch.int64)
        neg_pairs_dist_unique = mat[anchor_pairs_idx, neg_pairs_idx]

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
        self.num_negative_pairs = len(anchor_pairs_idx[neg_pairs_condition])
        indices_negative_pairs = (anchor_pairs_idx[neg_pairs_condition], neg_pairs_idx[neg_pairs_condition])
        # indices_negative_pairs = (anchor_pairs_idx, neg_pairs_idx)

        indices = (indices_triplets, indices_negative_pairs)
        return indices

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()
