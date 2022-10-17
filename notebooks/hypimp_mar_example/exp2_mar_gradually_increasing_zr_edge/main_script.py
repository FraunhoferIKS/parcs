from parcs.helpers.missing_data import indicator_graph_description_file, m_graph_convert
from parcs.graph_builder.randomizer import ConnectRandomizer, guideline_iterator
from parcs.cdag.graph_objects import Graph
import random as rand
import numpy as np
import pandas as pd
import json
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.utils.distributions import enable_reproducible_results
import warnings
warnings.filterwarnings("ignore")

# 0. configs
def RMSE(gt, ds, mask):
    se = np.square(
        gt[mask] - ds[mask]
    )
    mse = se.sum()/N
    return np.sqrt(mse)

data = pd.read_csv('./normalized_data.csv')
N = 700  # number of samples
N_total = len(data.columns) # number of total variables
N_O = 4  # number of fully observed variables
N_m = N_total - N_O  # number of missing feauters
miss_ratio = 0.5  # missing ratio in total

results = {}
for dir_, epoch, value in guideline_iterator(guideline_dir='guideline_2.yml',
                                             to_iterate='graph/graph_density',
                                             steps=5, repeat=10):
    print('GRAPH_DENSITY: {}, EPOCH: {}'.format(value, epoch))
    results[value] = {'hyperimpute': [], 'missforest': []}
    enable_reproducible_results(epoch)
    # 2. fully and partially observed variables
    obs_v = sorted(rand.sample(['Z_{}'.format(i) for i in range(N_total)], N_O))
    miss_v = sorted(list(set(data.columns) - set(obs_v)))
    total_v = sorted(obs_v + miss_v)
    # 3. write GDF for R
    indicator_graph_description_file(adj_matrix=np.zeros(shape=(N_m, N_m)),
                                     node_names=miss_v, miss_ratio=miss_ratio, subscript_only=True,
                                     file_dir='gdf_R.yml')
    # 4. randomize
    mask = pd.DataFrame(np.zeros(shape=(N_total, N_m)), index=total_v,
                        columns=['R_{}'.format(i.split('_')[1]) for i in miss_v])
    mask.loc[obs_v, :] = 1
    rndz = ConnectRandomizer(parent_graph_dir='../gdf_Z.yml', child_graph_dir='gdf_R.yml', guideline_dir=dir_,
                             adj_matrix_mask=mask)
    # 5. samples
    nodes, edges = rndz.get_graph_params()
    g = Graph(nodes=nodes, edges=edges)
    s = g.sample(N)
    r = ['R_{}'.format(i.split('_')[1]) for i in miss_v]
    print(s[r].mean())
    # outputs
    gt = s[total_v]
    ds = m_graph_convert(s, missingness_prefix='R_', shared_subscript=True)
    print(ds.isna().mean())
    raise

    # main thread
    mask = ds.isna().values
    mf = Imputers().get('missforest')
    hi = Imputers().get('hyperimpute')
    imp_hi = hi.fit_transform(ds)
    imp_mf = mf.fit_transform(ds)
    results[value]['hyperimpute'].append(RMSE(gt.values, imp_hi.values, mask))
    results[value]['missforest'].append(RMSE(gt.values, imp_mf.values, mask))
    del gt, ds, mask, mf, hi, imp_hi, imp_mf

with open('MAR_ZR_edge_density_variation.json', 'w') as f:
    json.dump(results, f)