from modules.sem.utils import is_acyclic
from modules.sem.graph_objects import BaseGraph
from modules.sem.graph_param import GraphParam
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class AutoEncoderSimulator:
    def __init__(self,
                 num_latent_vars=None,
                 num_nodes_in_hidden_layers=None,
                 output_layer_dtype_list=None,
                 randomization_config_dir=None):
        # defining node names
        latent_nodes = ['latent_{}'.format(i) for i in range(num_latent_vars)]
        hidden_layer_nodes = [
            ['hidden{}_{}'.format(num_hidden + 1, i + 1)
             for i in range(num_nodes_in_hidden_layers[num_hidden])]
            for num_hidden in range(len(num_nodes_in_hidden_layers))
        ]
        output_nodes = ['X{}'.format(i + 1) for i in range(len(output_layer_dtype_list))]
        total_nodes = latent_nodes + [item for sublist in hidden_layer_nodes for item in sublist] + output_nodes
        num_nodes = len(total_nodes)
        # defining node types
        node_types = {}
        for node in latent_nodes:
            node_types[node] = 'continuous'
        for layer in hidden_layer_nodes:
            for node in layer:
                node_types[node] = 'continuous'
        for node, dtype in zip(output_nodes, output_layer_dtype_list):
            node_types[node] = dtype
        # set adj matrix
        adj_matrix = pd.DataFrame(np.zeros(shape=(num_nodes, num_nodes)), columns=total_nodes, index=total_nodes)
        # == connect hidden to first layer
        adj_matrix.loc[latent_nodes, hidden_layer_nodes[0]] = 1
        # == interconnect hidden layers
        for i in range(len(hidden_layer_nodes) - 1):
            adj_matrix.loc[hidden_layer_nodes[i], hidden_layer_nodes[i + 1]] = 1
        # == last hidden layer to output
        adj_matrix.loc[hidden_layer_nodes[-1], output_nodes] = 1
        adj_matrix = adj_matrix.astype(int)

        self.graph_param = GraphParam().set_adj_matrix(
            set_type='custom', adj_matrix=adj_matrix
        ).set_node_output_types(
            set_type='custom', node_types_list=node_types
        ).set_edge_functions(
            set_type='full_random'
        ).set_edge_function_params(
            set_type='full_random', config_dir=randomization_config_dir
        ).set_state_functions(
            set_type='full_random'
        ).set_state_params(
            set_type='full_random', config_dir=randomization_config_dir
        ).set_output_functions(
            set_type='full_random'
        ).set_output_params(
            set_type='full_random', config_dir=randomization_config_dir
        )
        self.structure = BaseGraph().set_nodes(
            nodes_list=self.graph_param.get_node_list()
        ).set_edges(
            adj_matrix=self.graph_param.get_adj_matrix(),
            function_specs=self.graph_param.get_edge_function_specs()
        )

    def sample(self, size=None):
        self.structure.sample(size=size)
        return self

    def get_latent_vars(self):
        cols = [c for c in self.structure.data.columns if c[:6] == 'latent']
        return self.structure.data[cols]

    def get_output_vars(self):
        cols = [c for c in self.structure.data.columns if c[0] == 'X']
        return self.structure.data[cols]

    def get_full_vars(self):
        return self.structure.data


if __name__ == '__main__':
    ae = AutoEncoderSimulator(
        num_latent_vars=2,
        num_nodes_in_hidden_layers=[3],
        output_layer_dtype_list=['continuous'] * 4,
        randomization_config_dir='../../configs/params/default.yml'
    )
    ae.sample(size=500)
    data = ae.get_output_vars()
    plt.scatter(data.X1, data.X2, c=data.X3)
    plt.show()
