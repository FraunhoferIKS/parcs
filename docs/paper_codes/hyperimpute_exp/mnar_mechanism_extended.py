from pyparcs.helpers.missing_data import *
from pyparcs.graph_builder.randomizer import ConnectRandomizer
from pyparcs.cdag.graph_objects import Graph
from tqdm import tqdm
import random as rand
import pandas as pd
import numpy as np


# 0. configs
data = pd.read_csv('normalized_data.csv')
N = len(data)  # number of samples
N_total = len(data.columns) # number of total variables
miss_ratio = 0.5  # missing ratio in total
total_v = sorted(list(data.columns))
total_r = ['R_{}'.format(i.split('_')[1]) for i in total_v]

def get_miss_dataset():
    # 3. write GDF for R
    r_adj = R_adj_matrix(size=N_total, density=1)
    indicator_graph_description_file(adj_matrix=r_adj,
                                     node_names=total_v, miss_ratio=miss_ratio, subscript_only=True,
                                     file_dir='graph_description_files/gdf_R.yml')

    # 4. mask
    mask = pd.DataFrame(np.ones(shape=(N_total, N_total)), index=total_v, columns=total_r)
    rndz = ConnectRandomizer(parent_graph_dir='gdf_Z.yml', child_graph_dir='graph_description_files/gdf_R.yml',
                             guideline_dir='guidelines/guideline_1.yml',
                             adj_matrix_mask=mask)
    # 5. samples
    nodes, edges = rndz.get_graph_params()
    g = Graph(nodes=nodes, edges=edges)
    s = g.sample(full_data=True)
    # outputs
    gt = s[total_v]
    ds = m_graph_convert(s)

    return gt, ds[total_v]

def RMSE(gt, ds, mask):
    se = np.square(
        gt[mask] - ds[mask]
    )
    mse = se.sum()/mask.sum()
    return np.sqrt(mse)

# ====================================
import json
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.utils.distributions import enable_reproducible_results
import warnings
warnings.filterwarnings("ignore")

iters = 30
rmse_hi = []
rmse_mf = []
for it in tqdm(range(iters)):
    enable_reproducible_results(it)
    gt, ds = get_miss_dataset()
    mask = ds.isna().values
    mf = Imputers().get('missforest')
    hi = Imputers().get('hyperimpute')
    imp_hi = hi.fit_transform(ds)
    imp_mf = mf.fit_transform(ds)
    rmse_hi.append(RMSE(gt.values, imp_hi.values, mask))
    rmse_mf.append(RMSE(gt.values, imp_mf.values, mask))
    del gt, ds, mask, mf, hi, imp_hi, imp_mf

results = {'hyperimpute': rmse_hi, 'missforest': rmse_mf}
#
with open('results/MNAR_general_30.json', 'w') as f:
    json.dump(results, f)