import numpy as np
from pyparcs.helpers.missing_data import indicator_graph_description_file

indicator_graph_description_file(
    adj_matrix=np.zeros(shape=(4, 4)),
    node_names=['Z_{}'.format(i) for i in range(1, 5)],
    prefix='R',
    subscript_only=True,
    file_dir='./gdf_R.yml'
)
