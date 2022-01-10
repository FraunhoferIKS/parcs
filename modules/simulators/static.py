from modules.sem.utils import is_acyclic
from modules.sem.structures import SimpleStructure
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class AutoEncoderSimulator(SimpleStructure):
    def __init__(self,
                 num_latent_vars=None,
                 latent_nodes_adjm=None,
                 num_nodes_in_hidden_layers=None,
                 output_layer_dtype_list=None,
                 complexity=0):
        latent_nodes = ['latent_{}'.format(i) for i in range(num_latent_vars)]
        hidden_layer_nodes = [
            ['hidden{}_{}'.format(num_hidden + 1, i + 1)
             for i in range(num_nodes_in_hidden_layers[num_hidden])]
            for num_hidden in range(len(num_nodes_in_hidden_layers))
        ]
        output_nodes = ['x{}'.format(i + 1) for i in range(len(output_layer_dtype_list))]
        total_nodes = latent_nodes + [item for sublist in hidden_layer_nodes for item in sublist] + output_nodes
        num_nodes = len(total_nodes)

        adj_matrix = pd.DataFrame(np.zeros(shape=(num_nodes, num_nodes)), columns=total_nodes, index=total_nodes)

        if latent_nodes_adjm:
            assert is_acyclic(
                adj_matrix=pd.DataFrame(
                    latent_nodes_adjm, columns=latent_nodes, index=latent_nodes
                )
            )
            adj_matrix.loc[latent_nodes, latent_nodes] = latent_nodes_adjm

        # connect hidden to first layer
        adj_matrix.loc[latent_nodes, hidden_layer_nodes[0]] = 1
        # interconnect hidden layers
        for i in range(len(hidden_layer_nodes) - 1):
            adj_matrix.loc[hidden_layer_nodes[i], hidden_layer_nodes[i + 1]] = 1
        # last hidden layer to output
        adj_matrix.loc[hidden_layer_nodes[-1], output_nodes] = 1
        dtypes = ['continuous'] * (num_nodes - len(output_nodes)) + output_layer_dtype_list
        node_types = {name: type_ for name, type_ in zip(total_nodes, dtypes)}

        super().__init__(
            adj_matrix=adj_matrix,
            node_types=node_types,
            complexity=complexity
        )

    def get_latent_vars(self):
        cols = [c for c in self.data.columns if c[:6] == 'latent']
        return self.data[cols]

    def get_output_vars(self):
        cols = [c for c in self.data.columns if c[0] == 'x']
        return self.data[cols]

    def get_full_vars(self):
        return self.data


if __name__ == '__main__':
    a = AutoEncoderSimulator(
        num_latent_vars=2,
        num_nodes_in_hidden_layers=[2, 5, 10],
        output_layer_dtype_list=['continuous'] * 20,
        complexity=0
    )
    a.sample(size=1000)
    plt.scatter(a.data.x1, a.data.x2, c=a.data.x3)
    plt.show()
