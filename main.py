from modules.sem.structures import AutoEncoderSimulator
import numpy as np


ae = AutoEncoderSimulator(
    num_latent_vars=3,
    num_nodes_in_hidden_layers=[7, 10],
    latent_nodes_adjm=[[0, 1, 1], [0, 0, 1], [0, 0, 0]],
    output_layer_dtype_list=np.random.choice(['continuous', 'binary', 'categorical'], p=[0.99, 0.01, 0], size=20).tolist(),
    complexity=0
)

ae.sample(size=100)


# LATENT VARS
print(ae.get_latent_vars())
# OUTPUT VARS
print(ae.get_output_vars())
# FULL VARS
print(ae.get_full_vars())