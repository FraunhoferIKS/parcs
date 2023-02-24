[![DOI](https://zenodo.org/badge/592506885.svg)](https://zenodo.org/badge/latestdoi/592506885)
![PyPI](https://img.shields.io/pypi/v/pyparcs)

# PARCS: a Python Package for Causal Simulation

**PA**-rtially **R**-andomized **C**-ausal **S**-imulator is a simulation tool for causal 
methods. This library is designed to facilitate simulation study design and serve as a standard 
benchmarking tool for causal inference and discovery methods. PARCS generates simulation 
mechanisms based on causal DAGs and a wide range of adjustable parameters. Once the simulation 
setup is described via legible instructions and rules, PARCS automatically probes the space of 
all complying mechanisms and synthesizes data from both observational and interventional distributions. We encourage the causal inference researchers to utilize PARCS as a standard benchmarking tool for future works.

**_Funding statement:_** This project was funded by the Bavarian Ministry for Economic Affairs, 
Regional Development and 
 Energy as part of a project to support the thematic development of the Institute for Cognitive Systems.

**_Cite this work:_** The supporting research paper for PARCS will be announced here soon for 
citation and reference.

> **_NOTE:_** This project is under active development.

## Installation

Installation is possible using pip:

```commandline
pip install pyparcs
```

## Get started (A bare minimum)

To simulate a causal DAG, describe the graph in a _graph description file_:

```yaml
# === A causal Triangle: Treatment, Outcome, Confounder ===
# nodes
C: gaussian(mu_=0, sigma_=1)
A: gaussian(mu_=2C-1, sigma_=0.1C+1)
Y: gaussian(mu_=C+A-0.3AC, sigma_=2)
# edges
C->A: identity()
C->Y: identity()
A->Y: identity()
```

You can instantiate a graph object and sample from its observational and interventional 
distributions:

```python
from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
g.sample(size=5)
#           C         A         Y
# 0  1.500622  3.542066  3.928658
# 1  0.774417  2.115694  3.251244
# 2 -1.140551 -2.120171 -3.445699
# 3  0.590632  1.564428  0.109688
# 4 -0.652315 -2.649744 -6.378569

g.do(size=3, interventions={'A': 2.5})
#           C    A         Y
# 0 -1.047174  2.5  0.902704
# 1  0.099876  2.5  1.282226
# 2 -1.145309  2.5  3.391779

g.do_functional(size=3,
                intervene_on='Y', inputs=['A', 'C'],
                func=lambda a,c: (a+c)*10)
#           C         A          Y
# 0 -0.585768 -3.240235 -38.260031
# 1 -0.713663 -1.262177 -19.758394
# 2  1.925642  0.791920  27.175618
```

You can describe a graph partially and only up to a level:
```yaml
C: gaussian(mu_=1, sigma_=1)
A: gaussian(mu_=?, sigma_=1) # mu_ parameter is not specified
Y: random # Y conditional distribution is not specified

C->A: identity()
C->Y: identity()
A->Y: identity()
```

and let PARCS randomize the free parameters according to a guideline:
```yaml
nodes:
  bernoulli:
    p_: [ [f-range, 1, 2] , 0 , [f-range, 2, 3] ]
  gaussian:
    mu_: [ [f-range, -2, -1] , [f-range, 0.5, 1] , 0 ]
    sigma_: [ [f-range, 1, 3] , 0 , 0 ]
edges:
  identity: null
```

In this guideline, randomization ranges are specified (e.g. bias term for `mu_` is sampled 
from the continuous uniform `[-2, -1]`). 
```python
from pyparcs.graph_builder.randomizer import ParamRandomizer

rndz = ParamRandomizer(
    graph_dir='graph_description_1.yml',
    guideline_dir='simple_guideline.yml'
)
nodes, edges = rndz.get_graph_params()

g = Graph(nodes=nodes, edges=edges)
g.sample(size=3)
#           C         A    Y
# 0  1.660388  0.410814  1.0
# 1  1.253973 -2.983480  0.0
# 2  1.088486 -0.167692  1.0
```

## Documentation & Support

- The PARCS documentation can be found [here](https://fraunhoferiks.github.io/parcs/).
- Raise a development issue [here](https://github.com/FraunhoferIKS/parcs/issues)
- Contact the authors for theoretical and technical support:
  - [Alireza Zamanian](mailto:alireza.zamanian@iks.fraunhofer.de)
  - [Leopold Mareis](mailto:leopold.mareis@iks.fraunhofer.de)