from pyparcs.graph_builder.temporal_parsers import temporal


@temporal(['B', 'C'], 't-2')
def temporal_custom_func(data):
    return data['A'] + data['B_{t-2}'] + data['C_{t-1}']