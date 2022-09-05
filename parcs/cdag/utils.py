import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from itertools import combinations as comb


def topological_sort(adj_matrix: pd.DataFrame = None):
    try:
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
    except AssertionError:
        print('adj_matrix is not acyclic')
        raise

def is_adj_matrix_acyclic(adj_matrix):
    try:
        topological_sort(adj_matrix)
        return True
    except AssertionError:
        return False


def get_interactions(data):
    len_ = data.shape[1]
    return np.array([
        [np.prod(i) for r in range(2, len_ + 1) for i in comb(row, r)]
        for row in data
    ])

def get_interactions_length(len_):
    dummy_data = np.ones(shape=(len_,))
    return len([
        np.prod(i)
        for r in range(2, len_ + 1)
        for i in comb(dummy_data, r)
    ])

def get_interactions_dict(parents):
    len_ = len(parents)
    return [set(i) for r in range(2, len_ + 1) for i in comb(parents, r)]

def get_poly(data, n):
    return data ** n


def dot_prod(data, coef):
    if data.shape[0] == 0:
        data_augmented = [np.array([]) for _ in range(3)]
    else:
        # data_augmented = [data, get_poly(data, 2), get_interactions(data)]
        data_augmented = [data, get_interactions(data)]

    print(coef)
    print('===')

    return coef['bias'] + np.array([
        np.dot(d, coef[i])
        for (i, d) in zip(['linear', 'interactions'], data_augmented)
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
                # min - I =  6 -> I = min - 6
                # max - I = -6 -> I = max + 6
                error = np.inf
                theta = 0
                for i in np.linspace(array.min() - 6, array.max() + 6, 1000):
                    h = U * expit(array - i) + L
                    new_error = abs(h.mean() - self.target_mean)
                    if new_error <= error:
                        theta = i
                        error = new_error
                    else:
                        break
                self.config['offset'] = theta
            self.is_initialized = True

        return (self.config['upper'] - self.config['lower']) * \
               expit(array - self.config['offset']) + \
               self.config['lower']


class EdgeCorrection:
    def __init__(self, q_var=0.05):
        self.q_var = q_var
        self.is_initialized = False
        self.offset = None
        self.scale = None

    def transform(self, array):
        if not self.is_initialized:
            try:
                assert len(array) >= 500
            except AssertionError:
                warnings.warn(
                    """
                    PARCS calculate normalization statistics from the first input batch,
                    This is the 1st batch, while size < 500. It might lead to instabilities.
                    we recommend to run the first simulation run with greater size
                    """
                )
            # pick q quantiles
            array_trunc = np.sort(array)[int(len(array)*self.q_var): int(len(array)*(1-self.q_var))]
            self.offset = array_trunc.mean()
            self.scale = array_trunc.std()
            self.is_initialized = True

        return (array - self.offset) / self.scale