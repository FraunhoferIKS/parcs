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
from pyparcs.cdag.graph_objects import *
from pyparcs.graph_builder.parsers import *
from pyparcs.cdag.mapping_functions import *
import os
import pandas as pd


class TestSimpleGraph:
    @staticmethod
    def write_gdf(yml):
        with open('gdf.yml', 'w') as gdf:
            gdf.write(yml)

    @staticmethod
    def remove_gdf(dir_):
        os.remove(dir_)

    def test_graph_1(self):
        self.write_gdf(
            ("Z_1: gaussian(mu_=0, sigma_=1)\n"
             "Z_2: bernoulli(p_=0.3)\n"
             "Z_3: lognormal(mu_=Z_1+Z_2, sigma_=1)\n"
             "Z_1->Z_3: identity()\n"
             "Z_2->Z_3: sigmoid(alpha=1, beta=0, gamma=0, tau=1)")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        g = Graph(nodes, edges)
        samples, errors = g.sample(10, return_errors=True)

        # Manual graph
        z_1 = Node(name='Z_1',
                   output_distribution='gaussian',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [], 'interactions': []},
                       'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
                   do_correction=False
                   )
        z_2 = Node(name='Z_2',
                   output_distribution='bernoulli',
                   dist_params_coefs={
                       'p_': {'bias': 0.3, 'linear': [], 'interactions': []}},
                   do_correction=False
                   )
        z_3 = Node(name='Z_3',
                   output_distribution='lognormal',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 0]},
                       'sigma_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 0]}},
                   do_correction=False
                   )
        data = pd.DataFrame([], columns=('Z_1', 'Z_2', 'Z_3'))
        data['Z_1'] = z_1.calculate(data, [], errors['Z_1'])
        data['Z_2'] = z_2.calculate(data, [], errors['Z_2'])
        input_data = pd.DataFrame({
            'Z_1': edge_identity(data['Z_1'].values),
            'Z_2': edge_sigmoid(data['Z_2'].values, alpha=1, beta=0, gamma=0, tau=1)})
        data['Z_3'] = z_3.calculate(input_data, ['Z_1', 'Z_2'], errors['Z_3'])
        assert samples.equals(data)
        self.remove_gdf('gdf.yml')

    def test_graph_2(self):
        self.write_gdf(
            ("Z_1: gaussian(mu_=0, sigma_=1)\n"
             "Z_2: lognormal(mu_=3Z_1^2, sigma_=1)\n"
             "Z_1->Z_2: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        g = Graph(nodes, edges)
        samples, errors = g.sample(10, return_errors=True)

        # Manual graph
        z_1 = Node(name='Z_1',
                   output_distribution='gaussian',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [], 'interactions': []},
                       'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
                   do_correction=False
                   )
        z_2 = Node(name='Z_2',
                   output_distribution='lognormal',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [0], 'interactions': [3]},
                       'sigma_': {'bias': 1, 'linear': [0], 'interactions': [0]}},
                   do_correction=False
                   )
        data = pd.DataFrame([], columns=('Z_1', 'Z_2'))
        data['Z_1'] = z_1.calculate(data, [], errors['Z_1'])
        data['Z_2'] = z_2.calculate(data, ['Z_1'], errors['Z_2'])
        assert samples.equals(data)
        self.remove_gdf('gdf.yml')

    def test_graph_3(self):
        self.write_gdf(
            ("Z_1: gaussian(mu_=0, sigma_=1)\n"
             "Z_2: bernoulli(p_=Z_1^2), correction[]\n"
             "Z_3: exponential(lambda_=Z_1-Z_2), correction[lower=0, upper=10]\n"
             "Z_1->Z_2: identity()\n"
             "Z_1->Z_3: gaussian_rbf(alpha=1, beta=0, gamma=1, tau=2)\n"
             "Z_2->Z_3: arctan(alpha=1, beta=0, gamma=0)")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        g = Graph(nodes, edges)
        samples, errors = g.sample(10, return_errors=True)

        # Manual graph
        z_1 = Node(name='Z_1',
                   output_distribution='gaussian',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [], 'interactions': []},
                       'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
                   do_correction=False
                   )
        z_2 = Node(name='Z_2',
                   output_distribution='bernoulli',
                   dist_params_coefs={
                       'p_': {'bias': 0, 'linear': [0], 'interactions': [1]}},
                   do_correction=True,
                   correction_config={'lower': 0, 'upper': 1}
                   )
        z_3 = Node(name='Z_3',
                   output_distribution='exponential',
                   dist_params_coefs={
                       'lambda_': {'bias': 0, 'linear': [1, -1], 'interactions': [0, 0, 0]}},
                   do_correction=True,
                   correction_config={'lower': 0, 'upper': 10}
                   )
        data = pd.DataFrame([], columns=('Z_1', 'Z_2', 'Z_3'))
        data['Z_1'] = z_1.calculate(data, [], errors['Z_1'])
        data['Z_2'] = z_2.calculate(data, ['Z_1'], errors['Z_2'])
        input_data = pd.DataFrame({
            'Z_1': edge_gaussian_rbf(data['Z_1'].values, alpha=1, beta=0, gamma=1, tau=2),
            'Z_2': edge_arctan(data['Z_2'].values, alpha=1, beta=0, gamma=0)})
        data['Z_3'] = z_3.calculate(input_data, ['Z_1', 'Z_2'], errors['Z_3'])
        assert samples.equals(data)
        self.remove_gdf('gdf.yml')
