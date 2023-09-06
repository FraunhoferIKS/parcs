from pandas import DataFrame
from numpy import array


class GraphUtilsData:
    # tests get_adj_matrix()
    # inputs: node list, parents list, expected adjacency matrix
    get_adj_matrix = [
        # Normal scenario
        (['A', 'B', 'C'],
         {'A': [], 'B': ['A'], 'C': ['A', 'B']},
         DataFrame([[0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]],
                   index=['A', 'B', 'C'],
                   columns=['A', 'B', 'C']))
    ]

    # tests topological_sort()
    # inputs: adjacency matrix, expected sorted node list
    topological_sort = [
        (DataFrame([[0, 1, 1],
                   [0, 0, 1],
                   [0, 0, 0]],
                   index=['A', 'B', 'C'],
                   columns=['A', 'B', 'C']),
         ['A', 'B', 'C'])
    ]

    # tests get_interactions_values()
    # inputs: data, expected interactions data
    get_interactions_values = [
        (array([[1, 2, 3],
                [0, 0, 0],
                [1, 1, 1]]),
         array([[1, 2, 3, 4, 6, 9],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1]]))
    ]

    # tests get_interactions_len()
    # inputs: length of data, expected length of interactions
    get_interactions_length = [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 6)
    ]

    # tests get_interactions_names()
    # inputs: list of names, expected interaction terms for the names
    get_interactions_names = [
        ([], []),
        (['A'], [['A', 'A']]),
        (['A', 'B'], [['A', 'A'], ['A', 'B'], ['B', 'B']]),
    ]

    # tests dot_prod()
    # inputs: array, coefficients dict, expected result
    dot_prod = [
        (array([]),
         {'bias': 2.7, 'linear': [], 'interactions': []},
         2.7),
        (array([[0, 0],
                [1, 2],
                [10, 10]]),
         {'bias': 1, 'linear': [1, 10], 'interactions': [0, 1, 2]},
         array([1, 32, 411])),
        (array([[2], [1]]),
         {'bias': 0, 'linear': [1], 'interactions': [1]},
         array([6, 2])),
        (array([[2, 1],
                [1, 1]]),
         {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 2]},
         [5, 4]),
        (array([[1, 2],
                [3, 4]]),
         {'bias': 0, 'linear': [0, 0], 'interactions': [1, 1, 1]},
         [7, 37]),
    ]