from __future__ import absolute_import

from metric_learn import (ITML_Supervised, LMNN, LSML_Supervised,
                          SDML_Supervised, NCA, LFDA, RCA_Supervised)

from .euclidean import Euclidean
from .kissme import KISSME

factory = {
    'euclidean': Euclidean,
    'kissme': KISSME,
    'itml': ITML_Supervised,
    'lmnn': LMNN,
    'lsml': LSML_Supervised,
    'sdml': SDML_Supervised,
    'nca': NCA,
    'lfda': LFDA,
    'rca': RCA_Supervised,
}


def get_metric(algorithm):
    if algorithm not in factory:
        raise KeyError("Unknown metric:", algorithm)
    return factory[algorithm]
