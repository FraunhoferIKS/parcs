#  Copyright (c) 2023. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#  https://www.gnu.de/documents/gpl-2.0.de.html
#
#  Contact: alireza.zamanian@iks.fraunhofer.de
import os
import pytest
from pyparcs.graph_builder.randomizer import *


class TestParamRandomizer:
    @staticmethod
    @pytest.fixture(params=[
        ("A: random",
         "nodes:\n  bernoulli:\n    p_: [ [f-range, 0.1, 0.2] , 0 , 0 ]",
         {'output_distribution': 'bernoulli', 'bias': [0.1, 0.2], 'linear': [], 'interactions': []},
         [1, 0]),
        ("A: bernoulli(?)",
         "nodes:\n  bernoulli:\n    p_: [ [f-range, 0.8, 0.9] , 0 , 0 ]",
         {'output_distribution': 'bernoulli', 'bias': [0.8, 0.9], 'linear': [], 'interactions': []},
         [1, 0]),
        ("B: bernoulli(p_=0.2)\nA: bernoulli(?)\nB->A: identity()",
         "nodes:\n  bernoulli:\n    p_: [ [f-range, 0.1, 0.2] , [f-range, 0.2, 0.3] , [f-range, 0.4, 0.5] ]",
         {'output_distribution': 'bernoulli', 'bias': [0.1, 0.2], 'linear': [0.2, 0.3], 'interactions': [0.4, 0.5]},
         [2, 1]),
        ("B: bernoulli(p_=0.2)\nA: poisson(lambda_=?)\nB->A: identity()",
         "nodes:\n  poisson:\n    lambda_: [ [f-range, 0.1, 0.2] , [f-range, 0.2, 0.3] , [f-range, 0.4, 0.5] ]",
         {'output_distribution': 'poisson', 'bias': [0.1, 0.2], 'linear': [0.2, 0.3], 'interactions': [0.4, 0.5]},
         [2, 1])
    ])
    def files(request):
        # setup
        with open('gdf.yml', 'w') as f:
            f.write(request.param[0])
        with open('guideline.yml', 'w') as file:
            file.write(request.param[1])
        # test
        yield request.param[2], request.param[3]
        # teardown
        os.remove('guideline.yml')
        os.remove('gdf.yml')

    @staticmethod
    def test_node(files):
        # preliminaries
        rndz = ParamRandomizer(graph_dir='gdf.yml', guideline_dir='guideline.yml')
        nodes, edges = rndz.get_graph_params()
        # assert correct number of nodes have been established
        assert len(nodes) == files[1][0]
        assert len(edges) == files[1][1]
        # choose node A
        node = [n for n in nodes if n['name'] == 'A'][0]
        # check dist
        assert node['output_distribution'] == files[0]['output_distribution']
        # mark dist param for later addressing
        if node['output_distribution'] == 'bernoulli':
            p = 'p_'
        elif node['output_distribution'] == 'poisson':
            p = 'lambda_'
        else:
            raise AssertionError
        # check bias coef
        assert files[0]['bias'][0] <= node['dist_params_coefs'][p]['bias'] <= files[0]['bias'][1]
        # check linear and interaction coef if applicable
        if len(nodes) == 2:
            for coef in ['linear', 'interactions']:
                assert files[0][coef][0] <= node['dist_params_coefs'][p][coef] <= files[0][coef][1]

    @staticmethod
    @pytest.fixture(params=[
        ("A: bernoulli(p_=0.2)\nB: bernoulli(p_=0.5A)\nA->B: random",
         "edges:\n  arctan:\n    alpha: 2\n    beta: 0\n    gamma: [choice, 0, 1]",
         {'function_name': 'arctan', 'function_params': {'alpha': 2, 'beta': 0}}),
        ("A: bernoulli(p_=0.2)\nB: bernoulli(p_=0.5A)\nA->B: gaussian_rbf(alpha=?, beta=1, gamma=?, tau=2)",
         "edges:\n  gaussian_rbf:\n    alpha: 4\n    beta: 0\n    gamma: [choice, 0, 1]",
         {'function_name': 'gaussian_rbf', 'function_params': {'alpha': 4, 'beta': 1}}),
    ])
    def files_2(request):
        # setup
        with open('gdf.yml', 'w') as f:
            f.write(request.param[0])
        with open('guideline.yml', 'w') as file:
            file.write(request.param[1])
        # test
        yield request.param[2]
        # teardown
        os.remove('guideline.yml')
        os.remove('gdf.yml')

    @staticmethod
    def test_node(files_2):
        # preliminaries
        rndz = ParamRandomizer(graph_dir='gdf.yml', guideline_dir='guideline.yml')
        nodes, edges = rndz.get_graph_params()
        # assert correct number of nodes have been established
        assert len(nodes) == 2
        assert len(edges) == 1
        # assert edge is parsed
        edge = edges[0]
        assert edge['name'] == 'A->B'
        assert edge['function_name'] == files_2['function_name']
        assert edge['function_params']['alpha'] == files_2['function_params']['alpha']
        assert edge['function_params']['beta'] == files_2['function_params']['beta']
        assert edge['function_params']['gamma'] in {0, 1}
        if edge['function_name'] == 'gaussian_rbf':
            assert edge['function_params']['tau'] == 2
        elif edge['function_name'] == 'arctan':
            pass
        else:
            raise AssertionError
