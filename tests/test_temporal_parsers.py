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
from pyparcs.graph_builder.temporal_parsers import temporal_graph_file_parser
from pyparcs.cdag.graph_objects import Graph


class TestTemporalGraphFileParser:
    """
    This test class tests the parsing of the temporal description files.
    The plan is that a parsing function turns temporal gdfs into normal ones,
    so that the static parsers would take over from then on.
    """
    @staticmethod
    @pytest.fixture(params=[
        [("n_timesteps: 2\n"
          "A: bernoulli(p_=0.2)\n"
          "B_{t}: gaussian(mu_=2B_{t-1}+A, sigma_=1)\n"
          "B_{0}: poisson(lambda_=1.8)\n"
          "A->B_{t}: identity()\n"
          "B_{t-1}->B_{t}: identity()"),
         # nodes
         {'A': 'bernoulli', 'B_0': 'poisson', 'B_1': 'gaussian', 'B_2': 'gaussian'},
         # edges
         {'A->B_1': 'identity', 'A->B_2': 'identity', 'B_0->B_1': 'identity',
          'B_1->B_2': 'identity'}],
    ])
    def setup_gdf(request):
        # setup
        desc = request.param[0]
        nodes = request.param[1]
        edges = request.param[2]

        file_name = 'gdf_1.yml'
        with open(file_name, 'w') as f:
            f.write(desc)
        # test
        yield file_name, nodes, edges
        # teardown
        os.remove(file_name)

    @staticmethod
    def test_parses_gdf_correctly(setup_gdf):
        nodes, edges = temporal_graph_file_parser(setup_gdf[0])
        # Nodes
        # 1. names
        assert len(nodes) == len(setup_gdf[1])
        extracted_node_names = [i['name'] for i in nodes]
        assert set(extracted_node_names) == set(setup_gdf[1].keys())
        # 2. distributions
        for node in nodes:
            assert setup_gdf[1][node['name']] == node['output_distribution']

        # Edges
        assert len(edges) == len(setup_gdf[2])
        extracted_edge_names = [i['name'] for i in edges]
        assert set(extracted_edge_names) == set(setup_gdf[2].keys())
        # 2. distributions
        for edge in edges:
            assert setup_gdf[2][edge['name']] == edge['function_name']

    @staticmethod
    @pytest.fixture(scope='class')
    def det_node_required_files():
        # custom function
        with open('./customs_temp.py', 'w') as script:
            script.write("from pyparcs.graph_builder.temporal_parsers import temporal\n\n"
                         "@temporal(['B'],'t-1')\n"
                         "def custom_function(data): return data['A'] + data['B_{t-1}']")

        gdf = ("n_timesteps: 2\n"
               "A: bernoulli(p_=0.2)\n"
               "B_{t}: gaussian(mu_=2A, sigma_=1)\n"
               "B_{0}: poisson(lambda_=1.8)\n"
               "C_{t}: deterministic(customs_temp.py, custom_function)\n"
               "A->B_{t}: identity()\n"
               "A->C_{t}: identity()\n"
               "B_{t-1}->C_{t}: identity()")
        file_name = 'gdf_2.yml'
        with open(file_name, 'w') as f:
            f.write(gdf)
        # test
        yield file_name
        # teardown
        os.remove('./customs_temp.py')
        os.remove(file_name)

    @staticmethod
    def test_parse_deterministic_node(det_node_required_files):
        nodes, edges = temporal_graph_file_parser(det_node_required_files)
        graph = Graph(nodes, edges)
        samples = graph.sample(100)
        assert (samples['A'] + samples['B_0']).equals(samples['C_1'])
        assert (samples['A'] + samples['B_1']).equals(samples['C_2'])