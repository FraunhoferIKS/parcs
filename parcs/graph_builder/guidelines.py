def simple_guideline():
    return {
        'num_nodes': [4, 8],
        'adj_matrix_density': [0.5, 1],
        'output_distributions': {
            'bernoulli': {
            }
        }
    }


GUIDELINES = {
    'simple': simple_guideline()
}