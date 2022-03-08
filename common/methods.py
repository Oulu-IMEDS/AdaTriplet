import torch


class MetricLearningMethods(torch.nn.Module):
    def __init__(self, cfg, mining_func, loss_matching, loss_identity=None):
        super(MetricLearningMethods, self).__init__()
        self.cfg = cfg
        self.mining_function = mining_func
        self.loss_matching = loss_matching
        self.loss_identity = loss_identity
        self.list_dist = {}

    def get_no_triplets(self):
        num_triplets = self.mining_function.num_triplets
        if self.cfg.method in ['SCT', 'AdaTriplet', 'AdaTriplet-AM']:
            num_negative_pairs = self.mining_function.num_negative_pairs
        else:
            num_negative_pairs = 0
        if self.cfg.automargin_mode == 'ap':
            num_positive_pairs = self.mining_function.num_positive_pairs
        else:
            num_positive_pairs = 0
        return num_triplets, num_negative_pairs, num_positive_pairs

    def distance(self, f_a, f_p, f_n):
        if self.cfg.distance_loss == 'cosine':
            no_tripets = f_a.shape[0]
            no_features = f_a.shape[1]

            ap = torch.matmul(f_a.view(no_tripets, 1, no_features),
                              f_p.view(no_tripets, no_features, 1))
            an = torch.matmul(f_a.view(no_tripets, 1, no_features),
                              f_n.view(no_tripets, no_features, 1))
            d = an - ap + self.cfg.margin.m_loss

        elif self.cfg.distance_loss == 'l2':
            d_ap = torch.nn.functional.pairwise_distance(f_p, f_a, p=2)
            d_an = torch.nn.functional.pairwise_distance(f_n, f_a, p=2)
            d = d_ap - d_an + self.cfg.margin.m_loss
        else:
            raise ValueError(f'Not support distance type {self.cfg.distance_loss}')
        d = torch.squeeze(d)
        return d

    def distance_an(self, f_a, f_n):
        if self.cfg.distance_loss == 'cosine':
            no_tripets = f_a.shape[0]
            no_features = f_a.shape[1]
            an = torch.matmul(f_a.view(no_tripets, 1, no_features),
                              f_n.view(no_tripets, no_features, 1))
        elif self.cfg.distance_loss == 'l2':
            d_an = torch.nn.functional.pairwise_distance(f_n, f_a, p=2)
        else:
            raise ValueError(f'Not support diatance type {self.cfg.distance_loss}')
        d_an = torch.squeeze(an)
        return d_an

    def extract_regu_features(self, f_a, f_i, pair_type=None):
        no_tripets = f_a.shape[0]
        no_features = f_a.shape[1]
        if pair_type == 'negative':
            if self.cfg.method == 'AdaTriplet':
                beta_n = float(self.cfg.margin.beta)
            else:
                beta_n = self.auto_beta_n
            an = torch.matmul(f_a.view(no_tripets, 1, no_features),
                              f_i.view(no_tripets, no_features, 1))
            regu = an - beta_n
            embeddings_regu = torch.squeeze(regu).to(self.cfg.device)
        elif pair_type == 'positive':
            if self.cfg.method == 'ap':
                beta_p = float(self.cfg.margin.beta)
            else:
                beta_p = self.auto_beta_p
            ap = torch.matmul(f_a.view(no_tripets, 1, no_features),
                              f_i.view(no_tripets, no_features, 1))
            regu = beta_p - ap
            embeddings_regu = torch.squeeze(regu).to(self.cfg.device)

        return embeddings_regu

    def calculate_total_loss(self, embeddings, labels, epoch_id=-1, batch_id=-1):
        if self.cfg.method == 'Triplet-AM' or self.cfg.method == 'WAT':
            if batch_id == 0 and epoch_id > 0:
                dist = self.mining_function.get_ap_an_dists()
                self.dist = dist
            else:
                dist = 0
            self.mining_function.set_epoch_id_batch_id(epoch_id, batch_id)
            indices = self.mining_function(embeddings, labels)
            indices_tuple = indices[0]
            indices_negative_pairs = indices[1]
            indices_positive_pairs = indices[2]
            auto_margin = self.mining_function.get_margin()
            if not isinstance(auto_margin, torch.Tensor):
                auto_margin = torch.tensor(auto_margin, device=embeddings.device)
            self.loss_matching.set_margin(auto_margin)
        elif self.cfg.method == 'AdaTriplet-AM':
            if batch_id == 0:
                dist = self.mining_function.get_ap_an_dists()
            else:
                dist = 0
            self.mining_function.set_epoch_id_batch_id(epoch_id, batch_id)
            indices = self.mining_function(embeddings, labels)
            indices_tuple = indices[0]
            indices_negative_pairs = indices[1]
            indices_positive_pairs = indices[2]
            auto_margin = self.mining_function.get_margin()
            self.auto_beta_n = self.mining_function.get_beta_n()
            self.auto_beta_p = self.mining_function.get_beta_p()
            self.loss_matching.set_margin(auto_margin)
        elif self.cfg.method == 'Triplet' or self.cfg.method == 'AdaTriplet':
            indices = self.mining_function(embeddings, labels)
            indices_tuple = indices[0]
            indices_negative_pairs = indices[1]
            indices_positive_pairs = (torch.tensor([]), torch.tensor([]))
        elif self.cfg.method == 'SCT':
            indices = self.mining_function(embeddings, labels)
            indices_tuple = indices[0]
            indices_negative_pairs = indices[1]
            indices_positive_pairs = (torch.tensor([]), torch.tensor([]))
        else:
            indices_tuple = self.mining_function(embeddings, labels)

        f_anchor = embeddings[indices_tuple[0]]
        f_pos = embeddings[indices_tuple[1]]
        f_neg = embeddings[indices_tuple[2]]
        if self.loss_identity is not None:
            if len(indices_negative_pairs[0]) > 0 and len(indices_positive_pairs[0]) == 0:
                f_anchor_neg_pairs = embeddings[indices_negative_pairs[0]]
                f_neg_pairs = embeddings[indices_negative_pairs[1]]
                embeddings_regu = self.extract_regu_features(f_anchor_neg_pairs, f_neg_pairs, pair_type='negative')
                loss_neg = self.loss_identity(embeddings_regu)
                loss_pos = 0
            elif len(indices_negative_pairs[0]) == 0 and len(indices_positive_pairs[0]) > 0:
                f_anchor_pos_pairs = embeddings[indices_positive_pairs[0]]
                f_pos_pairs = embeddings[indices_positive_pairs[1]]
                embeddings_regu = self.extract_regu_features(f_anchor_pos_pairs, f_pos_pairs, pair_type='positive')
                loss_pos = self.loss_identity(embeddings_regu)
                loss_neg = 0
            elif len(indices_negative_pairs[0]) > 0 and len(indices_positive_pairs[0]) > 0:
                f_anchor_neg_pairs = embeddings[indices_negative_pairs[0]]
                f_neg_pairs = embeddings[indices_negative_pairs[1]]
                neg_embeddings_regu = self.extract_regu_features(f_anchor_neg_pairs, f_neg_pairs, pair_type='negative')
                loss_neg = 2 * self.loss_identity(neg_embeddings_regu)

                f_anchor_pos_pairs = embeddings[indices_positive_pairs[0]]
                f_pos_pairs = embeddings[indices_positive_pairs[1]]
                pos_embeddings_regu = self.extract_regu_features(f_anchor_pos_pairs, f_pos_pairs, pair_type='positive')
                loss_pos = self.loss_identity(pos_embeddings_regu)
            else:
                loss_neg = 0
                loss_pos = 0
            loss_id = self.cfg.loss.w_neg * loss_neg + loss_pos
        else:
            loss_id = 0

        if self.cfg.method == 'SCT':
            dist_mat = self.distance(f_anchor, f_pos, f_neg)
            loss_triplet = self.loss_matching(dist_mat, return_mean=False)
            f_anchor_neg_pairs = embeddings[indices_negative_pairs[0]]
            f_neg_pairs = embeddings[indices_negative_pairs[1]]
            dist_mat_an = self.distance_an(f_anchor_neg_pairs, f_neg_pairs)
            loss_pairs = self.cfg.loss.w_neg * dist_mat_an
            loss_matching = torch.mean(loss_triplet + loss_pairs)
        else:
            loss_matching = self.loss_matching(embeddings, labels, indices_tuple)

        loss = loss_matching + self.cfg.loss.w_lambda * loss_id

        return loss
