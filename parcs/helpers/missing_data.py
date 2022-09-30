import pandas as pd
import numpy as np
from itertools import combinations



def m_graph_convert(data: pd.DataFrame, missingness_prefix='R_', indicator_is_missed=0, shared_subscript=True):
    temp_data = data.copy(deep=True)
    len_prefix = len(missingness_prefix)
    # take Rs: it starts with prefix, and subtracting the prefix gives the name of another node
    r_columns = [
        i for i in data.columns if i[:len_prefix] == missingness_prefix
    ]
    if shared_subscript:
        r_subs = [r[len_prefix:] for r in r_columns]
        z_prefix = [z.split('_')[0] for z in data.columns]
        assert len(set(z_prefix) - {'R'}) == 1
        z_prefix = z_prefix[0]
        z_columns = ['{}_{}'.format(z_prefix, r) for r in r_subs]
    else:
        z_columns = [r[len_prefix:] for r in r_columns]
    assert len(set(z_columns).intersection(set(data.columns))) != 0

    # masking
    for z, r in zip(z_columns, r_columns):
        temp_data[z][temp_data[r] == indicator_is_missed] = np.nan

    return temp_data[set(data.columns) - set(r_columns)]


def nsc_mask(size=None):
    return np.ones(shape=(size, size)) - np.identity(size)


def sc_mask(size=None):
    return np.identity(size)


def block_conditional_mask(size=None):
    return np.triu(np.ones(shape=(size, size)), k=1)


def fully_observed_mar(shape=None, fully_observed_indices=None):
    assert len(fully_observed_indices) == shape[0] - shape[1]
    mask = np.zeros(shape=shape)
    mask[fully_observed_indices,:] = 1


def partially_observed_mar(size=None, fully_observed_indices=None):
    mask = np.ones(shape=(size, size))
    mask[fully_observed_indices, :] = 0
    mask[:, fully_observed_indices] = 0
    return mask

def R_adj_matrix(size=None, shuffle=False, density=1.0):
    adj_matrix = np.triu(np.ones(shape=(size, size)), k=1)
    if density < 1.0:
        s = np.random.choice([0, 1], p=[1-density, density], size=(size, size))
        adj_matrix = np.multiply(s, adj_matrix)
    if shuffle:
        inds = np.random.shuffle(range(size))
        adj_matrix = adj_matrix[inds, :][:, inds]
    return adj_matrix

def R_attrition_adj_matrix(size=None, step=None, density=1.0):
    adj_matrix = R_adj_matrix(size=size, shuffle=False, density=density)
    adj_matrix = np.multiply(
        adj_matrix,
        np.tril(np.ones(shape=(size, size)), k=step)
    )
    return adj_matrix

def indicator_graph_description_file(adj_matrix=None, node_names=None, prefix='R', subscript_only=False,
                                     file_dir=None):
    if subscript_only:
        sub = [n.split('_')[1] for n in node_names]
    else:
        sub = node_names
    r_names = ['{}_{}'.format(prefix, s) for s in sub]
    adj_matrix = pd.DataFrame(adj_matrix, index=r_names, columns=r_names)

    file = '# nodes\n'
    for r in r_names:
        file += '{}: bernoulli(p_=?), correction[]\n'.format(r)
    file += '# edges\n'
    for r1, r2 in combinations(r_names, 2):
        if adj_matrix.loc[r1, r2] == 1:
            file += '{}->{}: free\n'.format(r1, r2)
        elif adj_matrix.loc[r2, r1] == 1:
            file += '{}->{}: free\n'.format(r2, r1)
    with open(file_dir, 'w') as gdf:
        gdf.write(file)
    return None

