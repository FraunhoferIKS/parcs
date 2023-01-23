#  Copyright (c) 2022. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#  https://www.gnu.de/documents/gpl-2.0.de.html
#
#  Contact: alireza.zamanian@iks.fraunhofer.de

import pandas as pd
import numpy as np
from itertools import combinations
from pyparcs.exceptions import parcs_assert, DescriptionFileError


def m_graph_convert(data: pd.DataFrame, missingness_prefix='R', indicator_is_missed=0, shared_subscript=True):
    """
    This helper functions converts the synthesized data set of fully-observed variables and missingness indicators
    into a standard observational variable with missing entries.

    The function expects either one of the following naming styles for the columns:
        1. name of variables: XXX -> name of miss. indicators: <prefix>_XXX
        2. name of variables: X_Y -> name of miss. indicators: <prefix>_Y
    First option is suitable when you have distinct var names such as: "A, B, ..." or "age, sex, ..."
    Second option is suitable when you have single var name with numbers such as: "Z_1, Z_2, ..."

    Parameters
    ----------
    data : pd.DataFrame
        The PARCS-synthesized data of variables and miss. indicators
    missingness_prefix : str, default='R'
        prefix for miss. indicators
    indicator_is_missed : {0, 1}, default=0
        the value of miss. indicator which corresponds to masking the variable. e.g. if `0`, then
        `Z_1` is `NaN` if `R_1=0`.
    shared_subscript : bool, default=True
        to choose the naming style. if `True`, then {Z_1, R_1}. If `False` then {A, R_A}

    Returns
    -------
    masked_dataset : pd.DataFrame

    Raises
    ------
    DescriptionFileError
        If the prefix ends with underscore. The function automatically assumes the underscore.
    ValueError
        If missingness indicators are not binary (0, 1)

    """
    parcs_assert(
        missingness_prefix[-1] != '_',
        DescriptionFileError,
        '''
        missing prefix should not end with underscore _.
        The underscore for connecting prefix and subscripts will be considered automatically.
        '''
    )

    temp_data = data.copy(deep=True)
    len_prefix = len(missingness_prefix)
    # take Rs: it starts with prefix, and subtracting the prefix gives the name of another node
    r_columns = [
        i for i in data.columns if (
           i[:len_prefix] == missingness_prefix and  # starts with prefix and underscore
           '{}_{}'.format(missingness_prefix, i) not in data.columns  # this prevents from bug for words starting with R
        )
    ]
    r_indices = [r.split('_')[1] for r in r_columns]
    z_columns = list(set(data.columns) - set(r_columns))

    if shared_subscript:
        # there must be one z prefix
        z_prefix = [z.split('_')[0] for z in z_columns]
        assert len(set(z_prefix)) == 1
        z_columns = [z for z in z_columns if z.split('_')[1] in r_indices]
        r_columns = sorted(r_columns, key=lambda x: x.split('_')[1])
        z_columns = sorted(z_columns, key=lambda x: x.split('_')[1])
    else:
        z_columns = r_indices
    assert len(set(z_columns).intersection(set(data.columns))) != 0

    # check if r_columns are binary
    for r in r_columns:
        parcs_assert(
            set(temp_data[r].unique()) == {0, 1},
            ValueError,
            "Miss. indicator column {} is not binary.".format(r)
        )
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


def R_adj_matrix(size=None, shuffle=False, density=1.0):
    adj_matrix = np.triu(np.ones(shape=(size, size)), k=1)
    s = np.random.choice([0, 1], p=[1-density, density], size=(size, size))
    adj_matrix = np.multiply(s, adj_matrix)
    if shuffle:
        inds = [i for i in range(size)]
        np.random.shuffle(inds)
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
                                     file_dir=None, miss_ratio=None, supress_asteriks=False):
    if subscript_only:
        sub = [n.split('_')[1] for n in node_names]
    else:
        sub = node_names
    r_names = ['{}_{}'.format(prefix, s) for s in sub]
    adj_matrix = pd.DataFrame(adj_matrix, index=r_names, columns=r_names)

    if miss_ratio is None:
        ratio = ''
    else:
        assert 0 < miss_ratio < 1
        ratio = 'target_mean={}'.format(1-miss_ratio)
    if supress_asteriks:
        asteriks = ''
    else:
        asteriks = '*'
    file = '# nodes\n'
    for r in r_names:
        file += '{r}: bernoulli({a}p_=?), correction[{ratio}]\n'.format(r=r, a=asteriks, ratio=ratio)
    file += '# edges\n'
    for r1, r2 in combinations(r_names, 2):
        if adj_matrix.loc[r1, r2] == 1:
            file += '{}->{}: random\n'.format(r1, r2)
        elif adj_matrix.loc[r2, r1] == 1:
            file += '{}->{}: random\n'.format(r2, r1)
    with open(file_dir, 'w') as gdf:
        gdf.write(file)
    return None


