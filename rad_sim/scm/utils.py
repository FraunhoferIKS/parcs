import numpy as np
import pandas as pd
from scipy.special import expit
from itertools import combinations as comb


def topological_sort(adj_matrix: pd.DataFrame = None):
    adjm = adj_matrix.copy(deep=True).values
    ordered_list = []
    covered_nodes = 0
    while covered_nodes < adjm.shape[0]:
        # sum r/x -> r edges
        sum_c = adjm.sum(axis=0)
        # find nodes with no parents
        parent_inds = list(np.where(sum_c == 0)[0])
        assert len(parent_inds) != 0

        covered_nodes += len(parent_inds)
        # add to the list
        ordered_list += parent_inds
        # remove parent edges
        adjm[parent_inds, :] = 0
        # eliminate from columns by assigning values
        adjm[:, parent_inds] = 10
    return [adj_matrix.columns.tolist()[idx] for idx in ordered_list]


def get_interactions(data):
    len_ = data.shape[1]
    return np.array([
        [np.prod(i) for r in range(2, len_ + 1) for i in comb(row, r)]
        for row in data
    ])


def get_poly(data, n):
    return data ** n


def dot_prod(data, coef):
    if data.shape[0] == 0:
        data_augmented = [np.array([]) for _ in range(3)]
    else:
        data_augmented = [data, get_poly(data, 2), get_interactions(data)]

    return coef['bias'] + np.array([
        np.dot(d, coef[i])
        for (i, d) in zip(['linear', 'poly2', 'interactions'], data_augmented)
    ]).sum(axis=0)


class SigmoidCorrection:
    def __init__(self, lower=0, upper=1, target_mean=None, to_center=True):
        assert upper > lower
        if target_mean is not None:
            assert lower < target_mean < upper
        self.config = {
            'lower': lower,
            'upper': upper,
            'offset': 0,
            'offset_for_target_mean': 0
        }
        self.is_initialized = False
        self.to_center = to_center
        self.target_mean = target_mean

    def transform(self, array):
        if not self.is_initialized:
            # center
            if self.to_center:
                self.config['offset'] = array.mean()

            # transform by sigmoid
            U = (self.config['upper'] - self.config['lower'])
            L = self.config['lower']
            # fixing the mean via another offset (find by GD)
            if self.target_mean is not None:
                theta = array.mean()
                lr = 10
                target = (self.target_mean - L) / U
                for i in range(10000):
                    h = expit(array - theta)
                    delta = (h.mean() - target) * (h * (1 - h)).mean() * lr
                    theta += delta
                self.config['offset_for_target_mean'] = theta
            self.is_initialized = True

        return (self.config['upper'] - self.config['lower']) * \
               expit(array - self.config['offset'] - self.config['offset_for_target_mean']) + \
               self.config['lower']
