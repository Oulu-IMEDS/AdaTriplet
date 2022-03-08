from __future__ import absolute_import

import torch

from common.evaluators import extract_features
from common.metric_learning import get_metric, factory


class DistanceMetric(object):
    def __init__(self, algorithm='euclidean'):
        super(DistanceMetric, self).__init__()
        self.algorithm = algorithm
        if algorithm in factory:
            self.metric = get_metric(algorithm)

    def train(self, model, data_loader):
        if self.algorithm == 'euclidean' or self.algorithm == 'cosine': return
        features, labels = extract_features(model, data_loader)
        features = torch.stack(features.values()).numpy()
        labels = torch.Tensor(list(labels.values())).numpy()
        self.metric.fit(features, labels)

    def transform(self, X):
        if torch.is_tensor(X):
            X = X.cpu().detach().numpy()
            X = self.metric.transform(self.metric, X)
            X = torch.from_numpy(X)
        else:
            X = self.metric.transform(X)
        return X

