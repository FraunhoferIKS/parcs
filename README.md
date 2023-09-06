<h1 align="center">
<img src="https://raw.githubusercontent.com/FraunhoferIKS/parcs/9027c844fb1a46cacfdc55af5f54bf090ba8f707/images/parcs_light.svg" width="300">
</h1><br>



[![DOI](https://zenodo.org/badge/592506885.svg)](https://zenodo.org/badge/latestdoi/592506885)
![PyPI](https://img.shields.io/pypi/v/pyparcs)

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

To simulate a causal DAG, describe the graph by its nodes and edges and instantiate a graph object. You can sample from the graph's observational and interventional 
distributions:

```python
from pyparcs import Description, Graph
import numpy as np
np.random.seed(2023)

description = Description({'C': 'normal(mu_=0, sigma_=1)',
                           'A': 'normal(mu_=2C-1, sigma_=C^2+1)',
                           'Y': 'uniform(mu_=A+C, diff_=2)'},
                          infer_edges=True)
graph = Graph(description)
samples, _ = graph.sample(size=5)
#           C         A         Y
# 0  1.228778  0.297618  1.702500
# 1 -1.074313 -5.610021 -6.748542
# 2  0.604591 -2.538791 -1.885425
# 3 -0.109575 -1.104919 -1.211730
# 4 -1.031419 -3.615304 -4.924973

samples, _ = graph.do(size=3, interventions={'A': 2.5})
#           C    A         Y
# 0 -0.418041  2.5  1.442606
# 1 -1.803585  2.5  0.826138
# 2 -0.466009  2.5  1.787118

samples, _ = graph.do_functional(size=3,
                                 intervene_on='Y', inputs=['A', 'C'],
                                 func=lambda a, c: (a+c)*10)
#           C         A          Y
# 0 -1.259351 -5.846128 -71.054782
# 1 -0.309356 -2.557167 -28.665228
# 2  0.741366  1.578032  23.193976
```

You can describe a graph partially and only up to a level:
```yaml
# description_outline.yml

C: normal(mu_=1, sigma_=1)
A: normal(mu_=?C+2, sigma_=1) # mu_ parameter is not specified
Y: random # Y conditional distribution is not specified

C->A: identity()
C->Y: identity()
A->Y: identity()
```

and let PARCS randomize the free parameters according to a guideline:
```yaml
# guideline_outline.yml

nodes:
  bernoulli:
    p_: [ [f-range, 1, 2] , 0 , [f-range, 2, 3] ]
  normal:
    mu_: [ [f-range, -2, -1] , [f-range, 0.5, 1] , 0 ]
    sigma_: [ [f-range, 1, 3] , 0 , 0 ]
edges:
  identity: null
```

In this guideline, randomization ranges are specified (e.g. bias term for `mu_` is sampled 
from the continuous uniform `[-2, -1]`). 
```python
from pyparcs import Description, Graph, Guideline
import numpy as np
np.random.seed(2023)


description = Description('description_outline.yml')
guideline = Guideline('guideline_outline.yml')

description.randomize_parameters(guideline)
graph = Graph(description)
samples, _ = graph.sample(size=3)
#           C         A    Y
# 0  1.434386  1.447402  1.0
# 1  0.351719  2.092142  1.0
# 2 -0.026576  1.479390  1.0

# Randomized description for the graph
description.outline
# {'A': 'normal(mu_=1.0+0.66C, sigma_=1.0+C^2)',
#  'C': 'normal(mu_=1, sigma_=1.0)',
#  'C->A': 'identity()',
#  'C->Y': 'identity(), correction[]',
#  'Y': 'bernoulli(p_=1.44+2.67C^2), correction[]'}

```

## Documentation & Support

- The PARCS documentation can be found [here](https://fraunhoferiks.github.io/parcs/).
- Raise a development issue [here](https://github.com/FraunhoferIKS/parcs/issues)
- Contact the authors for theoretical and technical support:
  - [Alireza Zamanian](mailto:alireza.zamanian@iks.fraunhofer.de)
  - [Ruijie Chen](mailto:ruijie.chen@iks.fraunhofer.de)
  - [Leopold Mareis](mailto:leopold.mareis@iks.fraunhofer.de)