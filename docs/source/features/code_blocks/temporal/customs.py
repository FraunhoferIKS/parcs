from pyparcs.temporal import temporal_deterministic


@temporal_deterministic(['B', 'C'], 't-2')
def temporal_custom_func(data):
    return data['A'] + data['B_{t-2}'] + data['C_{t-1}']