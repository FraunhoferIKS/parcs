import numpy as np
from matplotlib import pyplot as plt
from rad_sim.sem.basic import IndependentUniformLatents, LatentLabelMaker
from rad_sim.simulators.temporal.deterministic import FourierSeries

sample_size = 200

# Fourier config
num_sins = 10
exp_decay = 1
seq_len = 300
frequency_range = [np.pi/seq_len, np.pi/6]
phaseshift_range = [-np.pi/4, np.pi/4]


var_list = [
    {'name': 'w_{}'.format(i), 'low': frequency_range[0], 'high': frequency_range[1]}
    for i in range(num_sins)
] + [
    {'name': 'phi_{}'.format(i), 'low': phaseshift_range[0], 'high': phaseshift_range[1]}
    for i in range(num_sins)
]


# === SAMPLE LATENTS ===
iul = IndependentUniformLatents()
iul.set_nodes(var_list=var_list)
latents = iul.sample(sample_size=sample_size)

# === SAMPLE FOURIER SERIES ===
fs = FourierSeries(
    sampled_latents=latents,
    dominant_amplitude=100,
    amplitude_exp_decay_rate=exp_decay,
    frequency_prefix='w',
    phaseshift_prefix='phi',
    added_noise_sigma=0.2
)
signals = fs.sample(seq_len=seq_len)

# === SAMPLE LABELS ===
lm = LatentLabelMaker(coef_min=1, coef_max=5)
y = lm.make_label(sampled_latents=latents[['w_{}'.format(i) for i in range(2)]])

for i in range(2):
    plt.plot(np.arange(seq_len), signals[i, :], c='blue' if y[i] == 1 else 'red')

plt.show()