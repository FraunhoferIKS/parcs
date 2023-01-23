from pyparcs.helpers.missing_data import indicator_graph_description_file, m_graph_convert, R_adj_matrix
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
N_O = 4  # number of fully observed variables
N_m = N_total - N_O  # number of missing features
miss_ratio = 0.5  # missing ratio in total

def get_miss_dataset():
    # 2. fully and partially observed variables
    obs_v = sorted(rand.sample(['Z_{}'.format(i) for i in range(N_total)], N_O))
    miss_v = sorted(list(set(data.columns) - set(obs_v)))
    total_v = sorted(obs_v + miss_v)
    # 3. write GDF for R
    r_adj = R_adj_matrix(size=N_m, shuffle=True, density=0.0)
    indicator_graph_description_file(adj_matrix=r_adj,
                                     node_names=miss_v, miss_ratio=miss_ratio, subscript_only=True,
                                     file_dir='graph_description_files/gdf_R.yml')
    # 4. randomize
    mask = pd.DataFrame(np.zeros(shape=(N_total, N_m)), index=total_v,
                        columns=['R_{}'.format(i.split('_')[1]) for i in miss_v])
    mask.loc[obs_v, :] = 1

    rndz = ConnectRandomizer(parent_graph_dir='gdf_Z.yml', child_graph_dir='graph_description_files/gdf_R.yml',
                             guideline_dir='guidelines/guideline_nonlin.yml',
                             adj_matrix_mask=mask)
    # 5. samples
    nodes, edges = rndz.get_graph_params()
    g = Graph(nodes=nodes, edges=edges)
    # s = g.sample(N)
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

iters = 10
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
with open('results/MAR_nonlin.json', 'w') as f:
    json.dump(results, f)