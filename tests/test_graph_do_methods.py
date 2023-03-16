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
import numpy as np
import pytest
from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
from pyparcs.exceptions import GraphError


class TestGraphDoMethods:
    """
    This class tests the functionality of intervention methods (do, do_functional, do_self)
    """
    @staticmethod
    def write_gdf(yml):
        """
        Writes a temp YAML graph description file (gdf)
        Parameters
        ----------
        yml : str
            the contents of the gdf.
        """
        with open('gdf.yml', 'w') as gdf:
            gdf.write(yml)

    @staticmethod
    def write_custompy(script):
        """
        Writes a temp custom.py script with a function, to test for DetNodes
        Parameters
        ----------
        script : str
            str of the code to be put in the Python script
        """
        with open('custom.py', 'w') as cpy:
            cpy.write(script)

    @staticmethod
    def remove_gdf():
        """
        Removes the temp gdf file
        """
        os.remove('gdf.yml')

    @staticmethod
    def remove_custompy():
        """
        Removes the custom.py Python script
        """
        os.remove('custom.py')

    def test_do_method(self):
        """
        Test if the do method works in general
        """
        self.write_gdf(
            ("A: gaussian(mu_=0, sigma_=1)\n"
             "B: bernoulli(p_=0.3A), correction[]\n"
             "C: lognormal(mu_=A+B, sigma_=1)\n"
             "A->B: identity()\n"
             "B->C: identity()\n"
             "A->C: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        dag = Graph(nodes, edges)
        samples, errors = dag.sample(20, return_errors=True)
        do_samples = dag.do(interventions={'B': 1}, use_sampled_errors=True, sampled_errors=errors)
        assert np.array_equal(do_samples[['A']], samples[['A']])
        assert np.array_equal(do_samples['B'].values, np.ones(20))
        assert not np.array_equal(do_samples[['C']], samples[['C']])
        self.remove_gdf()

    def test_do_method_raise_none_size_and_sampled_errors(self):
        """
        Test if the do method raises a ValueError if none of `size` and `sampled_errors` are
        given by the user
        """
        self.write_gdf(
            ("A: gaussian(mu_=0, sigma_=1)\n"
             "B: bernoulli(p_=0.3A), correction[]\n"
             "C: lognormal(mu_=B, sigma_=1)\n"
             "A->B: identity()\n"
             "B->C: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        dag = Graph(nodes, edges)
        with pytest.raises(ValueError):
            dag.do(interventions={'B': 1})
        self.remove_gdf()

    def test_functional_do(self):
        """
        Tests if the do_functional works in general
        """
        self.write_gdf(
            ("A: gaussian(mu_=0, sigma_=1)\n"
             "B: bernoulli(p_=0.3A), correction[]\n"
             "C: lognormal(mu_=B, sigma_=1)\n"
             "A->B: identity()\n"
             "B->C: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        dag = Graph(nodes, edges)

        samples = dag.do_functional(size=10, intervene_on='C', inputs=['A', 'B'],
                                    func=lambda a, b: a+b)
        assert samples['C'].equals(samples['A'] + samples['B'])
        self.remove_gdf()

    def test_functional_do_same_level_nodes(self):
        """
        Tests whether the all the nodes with the same topological sorting index, can be
        parents of each other in the functional do intervention.
        """
        self.write_gdf(
            ("A: gaussian(mu_=0, sigma_=1)\n"
             "B: bernoulli(p_=0.3), correction[]\n"
             "C: lognormal(mu_=A+B, sigma_=1)\n"
             "A->C: identity()\n"
             "B->C: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        dag = Graph(nodes, edges)

        samples = dag.do_functional(size=10, intervene_on='A', inputs=['B'], func=lambda b: b+1)
        assert samples['A'].equals(samples['B'] + 1)

        samples = dag.do_functional(size=10, intervene_on='B', inputs=['A'], func=lambda a: a+1)
        assert samples['B'].equals(samples['A'] + 1)
        self.remove_gdf()

    def test_functional_do_raise_descendant_parents(self):
        """
        Tests if the do_functional method raises a Graph error when the selected inputs
        for a node are in descendants of that node
        """
        self.write_gdf(
            ("A: gaussian(mu_=0, sigma_=1)\n"
             "B: bernoulli(p_=0.3)\n"
             "C: lognormal(mu_=B, sigma_=1)\n"
             "A->B: identity()\n"
             "B->C: identity()")
        )
        # PARCS Graph
        nodes, edges = graph_file_parser('gdf.yml')
        dag = Graph(nodes, edges)
        with pytest.raises(GraphError):
            dag.do_functional(size=10, intervene_on='B', inputs=['C'], func=lambda c: c + 1)
        with pytest.raises(GraphError):
            dag.do_functional(size=10, intervene_on='A', inputs=['C'], func=lambda c: c + 1)
        self.remove_gdf()
