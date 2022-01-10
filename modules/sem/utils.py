import numpy as np
import pandas as pd


def exp_prob(complexity=None, num_categories=None):
    assert num_categories >= 1
    # exp( k.(c-0.5).(x-x0) )
    # x: 0->simplest option ... n->most complex option
    k = 5
    x_0 = 3
    P = [
        np.exp(k * (complexity - 0.5) * (x - x_0))
        for x in range(num_categories)
    ]
    return [p / sum(P) for p in P]


def mask_matrix(matrix=None, mask=None, mask_value=0):

    n = matrix.shape[0]
    return np.array([
        [
            matrix[i, j] if mask[i, j]
            else mask_value for j in range(n)
        ] for i in range(n)
    ])


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


def is_acyclic(adj_matrix=None):
    try:
        topological_sort(adj_matrix=adj_matrix)
        return True
    except AssertionError:
        return False


if __name__ == '__main__':
    import pandas as pd
    x = pd.DataFrame([[1, 1], [2, 0], [1, 0]], columns=('a','b'))
    x['c'] = x['a'].mask(x['b'] == 0, np.nan)
    print(x)
