from pyparcs.core.exceptions import *


class MiscData:
    # tests augment_line
    # inputs: line, term to add, limit, (is_added, expected line)
    augment_line_data = [
        # non-stochastic or random nodes must not change
        ('random', '2A', False,
         (False, 'random')),
        ('random, tags[P1]', '?A', False,
         (False, 'random,tags[P1]')),
        ('random, correction[]', '?A', False,
         (False, 'random,correction[]')),
        ('constant(2)', '?A', False,
         (False, 'constant(2)')),
        # normal scenario
        ('normal(mu_=A+B, sigma_=1)', '?C', False,
         (True, 'normal(mu_=A+B+?C,sigma_=1+?C)')),
        # limit applied
        ('lognormal(!mu_=A+B, sigma_=1)', '?C', True,
         (True, 'lognormal(mu_=A+B,sigma_=1+?C)')),
        ('uniform(!mu_=A+B, !diff_=1)', '?C', True,
         (False, 'uniform(mu_=A+B,diff_=1)')),
        # "!" with no limit
        ('bernoulli(!p_=A+B)', '?C', False,
         (True, 'bernoulli(p_=A+B+?C)')),
        # tags and correction
        ('bernoulli(p_=A+B), tags[P1], correction[]', '?C', False,
         (True, 'bernoulli(p_=A+B+?C),tags[P1],correction[]')),
        ('bernoulli(p_=A+B), correction[target_mean=0.3]', '?C', False,
         (True, 'bernoulli(p_=A+B+?C),correction[target_mean=0.3]')),
    ]

    # tests tag parser
    # inputs: line, set of expected tags
    tag_parser_data = [
        ('constant(2), tags[P1, D2]', {'P1', 'D2'}),
        ('bernoulli(!p_=A+B)', set()),
        ('constant(2), correction[], tags[C]', {'C'})
    ]

    # tests tag parser raises error
    # inputs: line
    tag_parser_erroneous_data = [
        'bernoulli(!p_=A+B), tags[]'
    ]


class TermParserData:
    # tests term parsers normal behavior
    # inputs: term, variables, expected parents, expected coefs
    term_parser_data = [
        # bias terms
        ('1.5', ['A', 'B'], [], 1.5),
        ('3', ['A', 'B'], [], 3),
        ('-0', ['A', 'B'], [], 0),
        ('-0.6', ['A', 'B'], [], -0.6),
        ('-2', ['A', 'B'], [], -2),
        # bias with no parent
        ('-2', [], [], -2),
        # linear terms
        ('B', ['A', 'B', 'B_1'], ['B'], 1),
        ('-0.3Z_1', ['A', 'B', 'Z_1'], ['Z_1'], -0.3),
        ('1.7A', ['A', 'B', 'Z_1'], ['A'], 1.7),
        # names which are identical up to a letter: Z_1, Z_11
        ('4Z_11', ['Z_1', 'Z_11', 'Z_19'], ['Z_11'], 4),
        ('3Z_1', ['Z_1', 'Z_11', 'Z_19'], ['Z_1'], 3),
        # interaction terms
        ('ABB_1', ['A', 'B', 'B_1'], ['A', 'B', 'B_1'], 1),
        ('2B_1A', ['A', 'B', 'B_1'], ['A', 'B_1'], 2),
        ('-0.3BA', ['A', 'B', 'B_1'], ['A', 'B'], -0.3),
        # names identical up to a letter: Z_1, Z_11
        ('4Z_11Z_1', ['Z_1', 'Z_11', 'Z_19'], ['Z_11', 'Z_1'], 4),
        # quadratic terms
        ('A^2', ['A', 'B', 'Z_1'], ['A', 'A'], 1),
        ('1.6Z_1^2', ['A', 'B', 'Z_1'], ['Z_1', 'Z_1'], 1.6),
        ('-3B^2', ['A', 'B', 'Z_1'], ['B', 'B'], -3),
        # names which are identical up to a letter: Z_1, Z_11
        ('Z_1^2', ['Z_1', 'Z_11', 'Z_19'], ['Z_1', 'Z_1'], 1),
        # '?' coefficients for partial descriptions
        ('?', ['A', 'B'], [], '?'),
        ('?B', ['A', 'B', 'B_1'], ['B'], '?'),
        ('?Z_11Z_1', ['Z_1', 'Z_11', 'Z_19'], ['Z_11', 'Z_1'], '?'),
        ('?Z_1^2', ['Z_1', 'Z_11', 'Z_19'], ['Z_1', 'Z_1'], '?')
    ]

    # tests correct error raising
    # inputs: term, variables, expected error
    term_parser_erroneous_data = [
        # bias terms
        ('J', ['A', 'B', 'C'], DescriptionError),  # not existing parent
        ('AZ_1A', ['A', 'B', 'C'], DescriptionError),  # parent duplicate
        ('AA', ['A', 'B', 'C'], DescriptionError),  # parent duplicate
        ('2B^2A', ['A', 'B', 'C'], DescriptionError),  # invalid quadratic term
        ('2AB^2', ['A', 'B', 'C'], DescriptionError),  # invalid quadratic term
        ('2B^3', ['A', 'B', 'C'], DescriptionError),  # invalid power
        ('-?B', ['A', 'B', 'C'], DescriptionError),  # negative question mark
    ]


class EquationParserData:
    # tests equation parser
    # inputs: equation, variables, expected output
    eq_parser = [
        # example equation, no space
        ('2+A-2.8B', ['A', 'B'], [([], 2), (['A'], 1.0), (['B'], -2.8)]),
        # only bias
        ('1', ['A', 'B'], [([], 1)]),
        # quadratic terms with space
        ('A^2-2AB', ['A', 'B'], [(['A', 'A'], 1), (['A', 'B'], -2.0)]),
        ('Z_1Z_11 + 2Z_11 - Z_1',
         ['Z_1', 'Z_11'],
         [(['Z_11', 'Z_1'], 1), (['Z_11'], 2.0), (['Z_1'], -1.0)]),
        # terms with question marks
        ('2+A+?B', ['A', 'B'], [([], 2), (['A'], 1.0), (['B'], '?')]),
        ('A^2+?B^2+?AB', ['A', 'B'], [(['A', 'A'], 1), (['B', 'B'], '?'), (['A', 'B'], '?')])
    ]

    # tests correct error raising
    # inputs: equation, variables, expected error
    eq_parser_erroneous_data = [
        # duplicate terms
        ('2A + 3A', ['A', 'B'], DescriptionError),
        ('2AB + 3BA', ['A', 'B'], DescriptionError),
        ('B + 2A^2 - A^2', ['A', 'B'], DescriptionError),
        # non-existing parents
        ('B + 2A^2', ['B'], DescriptionError),
        # non-standard symbols
        ('A + B * 3', ['B'], DescriptionError),

    ]


class NodeParserData:
    # tests constant node
    # inputs: line, parents, expected output dict
    const_node_data = [
        ('constant(2)', ['A', 'B'], {'value': 2, 'node_type': 'constant', 'tags': []}),
        ('constant(-0.3)', ['A', 'B'], {'value': -0.3, 'node_type': 'constant', 'tags': []}),
        ('constant(0)', ['A', 'B'], {'value': 0, 'node_type': 'constant', 'tags': []}),
    ]

    # tests constant node raises error
    # inputs: line, parents
    const_node_erroneous_data = [
        ('constant(A)', ['A', 'B']),  # no parents
        ('constant()', ['A', 'B']),  # empty const
    ]

    # tests stochastic node
    # inputs: line, parents, distribution, parameter coefs, do_correction, correction config
    stochastic_node_data = [
        ('bernoulli(p_=2A+B^2)', ['A', 'B'], 'bernoulli',
         {'p_': {'bias': 0, 'linear': [2, 0], 'interactions': [0, 0, 1]}}, False, {}),
        ('normal(mu_=1-0.3AB, sigma_=2)', ['A', 'B'], 'normal',
         {'mu_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, -0.3, 0]},
          'sigma_': {'bias': 2, 'linear': [0, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('uniform(mu_=4B, diff_=A^2)', ['A', 'B'], 'uniform',
         {'mu_': {'bias': 0, 'linear': [0, 4], 'interactions': [0, 0, 0]},
          'diff_': {'bias': 0, 'linear': [0, 0], 'interactions': [1, 0, 0]}}, False, {}),
        ('lognormal(mu_=A+B, sigma_=A)', ['A', 'B'], 'lognormal',
         {'mu_': {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 0]},
          'sigma_': {'bias': 0, 'linear': [1, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('poisson(lambda_=B^2+1)', ['A', 'B'], 'poisson',
         {'lambda_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 1]}}, False, {}),
        ('exponential(lambda_=-AB)', ['A', 'B'], 'exponential',
         {'lambda_': {'bias': 0, 'linear': [0, 0], 'interactions': [0, -1, 0]}}, False, {}),
        # parentless nodes: only test one distribution since logic is the same
        ('bernoulli(p_=2)', [], 'bernoulli',
         {'p_': {'bias': 2, 'linear': [], 'interactions': []}}, False, {}),
        # partially randomized cases
        ('bernoulli(?)', ['A'], 'bernoulli',
         {'p_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
        ('normal(mu_=?, sigma_=2A)', ['A', 'B'], 'normal',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': 0, 'linear': [2, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('normal(mu_=?, sigma_=?)', ['A', 'B'], 'normal',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
        ('lognormal(?)', ['A', 'B'], 'lognormal',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
        # partial coefficients
        ('bernoulli(p_=2A+?B^2+?)', ['A', 'B'], 'bernoulli',
         {'p_': {'bias': '?', 'linear': [2, 0], 'interactions': [0, 0, '?']}}, False, {}),
        ('normal(mu_=1-0.3AB+?A^2, sigma_=2+?A)', ['A', 'B'], 'normal',
         {'mu_': {'bias': 1, 'linear': [0, 0], 'interactions': ['?', -0.3, 0]},
          'sigma_': {'bias': 2, 'linear': ['?', 0], 'interactions': [0, 0, 0]}}, False, {}),

    ]

    # tests stochastic node raises error
    # inputs: line, parents
    stochastic_node_erroneous_data = [
        ('fakedist(p_=2A+B^2)', ['A', 'B']),  # wrong distribution name
        ('bernoulli(mu_=2A+B^2)', ['A', 'B']),  # wrong parameter name
        ('normal(mu_=2A+B^2, mu_=2, sigma_=3)', ['A', 'B']),  # duplicate params
        ('exponential(lambda_=2A+B^2)', []),  # wrong parents
        ('poisson(lambda_=B^2)', ['A']),  # wrong parents
        ('poisson(lambda_=B^2+?A)', ['A'])  # wrong randomization
    ]

    # tests do correction
    # inputs: line, parents, expected config
    do_correction_data = [
        ('bernoulli(p_=A), correction[target_mean=0.3, lower=0, upper=1]', ['A'],
         {'target_mean': 0.3, 'lower': 0, 'upper': 1}),
        ('bernoulli(p_=A), correction[]', ['A'], {})
    ]

    # tests do correction raises error
    # inputs: line, parents
    do_correction_erroneous_data = [
        ('bernoulli(p_=A), correction[lower=1, upper=2+X]', ['A']),  # non-float values
    ]


class EdgeParserData:
    # tests edge parser
    # inputs: line, function name, function parameters, do correction
    edge_parser_data = [
        # normal
        ('identity()', 'identity', {}, False),
        ('sigmoid(alpha=2, beta=1, gamma=0, tau=1)', 'sigmoid',
         {'alpha': 2, 'beta': 1, 'gamma': 0, 'tau': 1}, False),
        ('gaussian_rbf(alpha=2, beta=1, gamma=0, tau=1)', 'gaussian_rbf',
         {'alpha': 2, 'beta': 1, 'gamma': 0, 'tau': 1}, False),
        ('arctan(alpha=2, beta=1, gamma=0)', 'arctan',
         {'alpha': 2, 'beta': 1, 'gamma': 0}, False),
        # partially randomized
        ('sigmoid(alpha=?, beta=1, gamma=?, tau=1), correction[]', 'sigmoid',
         {'alpha': '?', 'beta': 1, 'gamma': '?', 'tau': 1}, True),
        ('gaussian_rbf(?)', 'gaussian_rbf',
         {'alpha': '?', 'beta': '?', 'gamma': '?', 'tau': '?'}, False)
    ]

    # tests edge parser error raising
    # inputs: line
    edge_parser_erroneous_data = [
        'fakeedge(alpha=1)',  # wrong edge function name
        'identity(alpha=1)',  # wrong params
        'gaussian_rbf(alpha=2, beta=1)',  # incomplete params
        'gaussian_rbf(alpha=2, beta=1, gamma=2, tau=A)',  # non-number params
    ]


class DescriptionParserData:
    # tests infer edges
    # inputs: nodes sublist, edges sublist, expected edges_sublist
    infer_edges_data = [
        # a normal routine scenario with no name complication, where an edge needs to be inferred
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'bernoulli(p_=0.4)',
          'C': 'normal(mu_=2A+B^2-1, sigma_=1)'},
         {},
         {'A->C': 'identity()', 'B->C': 'identity()'}),
        # a routine scenario with no name complication, where no implicit edge needs to be inferred
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'bernoulli(p_=0.4)',
          'C': 'normal(mu_=2A+B^2-1, sigma_=1)'},
         {'A->C': 'identity()', 'B->C': 'identity()'},
         {'A->C': 'identity()', 'B->C': 'identity()'}),
        # a scenario with name complication where the true edge will be inferred
        ({'A': 'bernoulli(p_=0.2)',
          'B': 'bernoulli(p_=0.3AB_2)',
          'B_2': 'bernoulli(p_=0.5)'},
         {},
         {'A->B': 'identity()', 'B_2->B': 'identity()'})
    ]

    # tests infer edges raises error
    # inputs: outline
    infer_edges_erroneous_data = [
        {'A': 'bernoulli(p_=0.1)', 'A_1': 'bernoulli(p_=0.4)', 'B': 'bernoulli(p_=2A_2)'},
        {'A': 'bernoulli(p_=0.1)', 'A_1': 'bernoulli(p_=0.4)', 'B': 'bernoulli(p_=2A2)'},
        {'A': 'bernoulli(p_=0.1)', 'A_1': 'bernoulli(p_=0.4)', 'B': 'bernoulli(p_=2A_2+1)'}
    ]


class SynthesizerData:
    # tests term synthesizer
    # inputs: parsed, expected synthesized
    term_synthesizer_data = [
        (([], 2), '2'),
        (([], 1.123123), '1.12'),
        (([], -1), '-1'),
        (([], 1), '1'),
        (([], 0), ''),
        (([], '?'), '?'),
        ((['A'], 1), 'A'),
        ((['A'], -1), '-A'),
        ((['A'], 0), ''),
        ((['A', 'B'], 1.2), '1.2AB'),
        ((['A', 'B'], -2.123), '-2.12AB'),
        ((['A', 'B'], 0), ''),
        ((['A', 'B'], '?'), '?AB'),
        ((['A', 'A'], 1.2), '1.2A^2'),
        ((['A', 'A'], -2.3), '-2.3A^2'),
        ((['A', 'A'], 0), ''),
        ((['A', 'A'], '?'), '?A^2')
    ]

    # tests equation synthesizer
    # inputs: parsed, expected synthesized
    equation_synthesizer_data = [
        ([([], 2), (['A', 'A'], 1)], '2+A^2'),
        ([(['A', 'A'], '?'), (['A', 'B'], -2.1)], '?A^2-2.1AB'),
        ([([], 0.0)], '0.0')
    ]

    # tests node synthesizer
    # inputs: node, parents, tags, expected synthesized'
    node_synthesizer_data = [
        ({'output_distribution': 'bernoulli',
          'dist_params_coefs': {'p_': {'bias': 0.2,
                                       'linear': [1, 2],
                                       'interactions': [0, 1, 0]}},
          'do_correction': True,
          'correction_config': {'upper': 0.5, 'target_mean': 0.2}},
         ['A', 'B'],
         ['P1', 'D2'],
         'bernoulli(p_=0.2+A+2B+AB), correction[upper=0.5, target_mean=0.2], tags[P1, D2]')
    ]

    # tests edge synthesizer
    # inputs: edge, tags, synthesized
    edge_synthesizer_data = [
        ({'function_name': 'sigmoid',
          'function_params': {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 3},
          'do_correction': True},
         ['P1', 'D2'],
         'sigmoid(alpha=1, beta=0, gamma=0, tau=3), correction[], tags[P1, D2]')
    ]