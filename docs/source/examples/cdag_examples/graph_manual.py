import numpy as np
import pandas as pd
from parcs.cdag.graph_objects import Node, Edge
np.random.seed(1)

# define nodes
z_1 = Node(
    name='Z_1',
    parents=[],
    output_distribution='gaussian',
    dist_params_coefs={
        'mu_': {'bias': 0, 'linear': np.array([]), 'interactions': np.array([])},
        'sigma_': {'bias': 3, 'linear': np.array([]), 'interactions': np.array([])},
    }
)
z_2 = Node(
    name='Z_2',
    parents=[],
    output_distribution='gaussian',
    dist_params_coefs={
        'mu_': {'bias': 0, 'linear': np.array([]), 'interactions': np.array([])},
        'sigma_': {'bias': 3, 'linear': np.array([]), 'interactions': np.array([])},
    }
)
z_3 = Node(
    name='Z_3',
    parents=['Z_1', 'Z_2'],
    output_distribution='bernoulli',
    dist_params_coefs={
        'p_': {'bias': 0, 'linear': np.array([0, 0]), 'interactions': np.array([1])}
    },
    do_correction=True,
    dist_configs={
        'correction_config': { 'p_': {'lower': 0, 'upper': 1} }
    }
)

# define edges
e_13 = Edge(
    parent='Z_1',
    child='Z_3',
    function_name='identity',
    function_params={}
)
e_23 = Edge(
    parent='Z_2',
    child='Z_3',
    function_name='identity',
    function_params={}
)

# sample nodes
data = pd.DataFrame([], columns=('Z_1', 'Z_2', 'Z_3'))
data['Z_1'] = z_1.sample(data, size=500)
data['Z_2'] = z_2.sample(data, size=500)
data['Z_3'] = z_3.sample(
    pd.DataFrame({
        'Z_1': e_13.map(data['Z_1'].values),
        'Z_2': e_23.map(data['Z_2'].values),
    })
)

print(data.head())
#         Z_1       Z_2  Z_3
# 0  4.873036 -5.158183    0
# 1 -1.835269  0.171363    0
# 2 -1.584515 -2.398642    1
# 3 -3.218906 -0.874784    1
# 4  2.596223 -0.776949    0