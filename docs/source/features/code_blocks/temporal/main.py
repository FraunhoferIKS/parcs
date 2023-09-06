from pyparcs import Graph
from pyparcs.temporal import TemporalDescription
import numpy as np
np.random.seed(42)

description = TemporalDescription('description.yml', n_timesteps=3)
graph = Graph(description)
samples, _ = graph.sample(size=5)

print(samples[['A', 'B_1', 'B_2', 'B_3', 'C_1', 'C_2', 'C_3']])
#           A       B_1       B_2       B_3       C_1       C_2       C_3
# 0  0.749080  2.618855  2.868731  1.857774  1.749080  4.498160  7.866095
# 1  0.041169  2.963863  2.165531  1.257099  1.041169  3.082338  6.087370
# 2  1.223706  1.452870  1.111365  1.001025  2.223706  5.447412  8.123987
# 3  1.215090  0.486305  2.120447  3.940599  2.215090  5.430179  7.131574
# 4  0.244076  0.180118  1.516700  0.869589  1.244076  3.488153  3.912348

print((samples['C_1']).equals(samples['A'] + samples['B_neg1'] + samples['C_0']))
print((samples['C_2']).equals(samples['A'] + samples['B_0'] + samples['C_1']))
print((samples['C_3']).equals(samples['A'] + samples['B_1'] + samples['C_2']))
# True
# True
# True
