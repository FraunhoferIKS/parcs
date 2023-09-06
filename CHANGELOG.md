# V-1.0.0 changelog

V1.0.0 introduces a new structure in PARCS, where declaration and randomization of the DAGs can be done easier and more effectively.

## New Structure

- **Description objects**: `pyparcs.Description` class handles the functionalities about declaring and randomization of DAGs. Parameter and connection randomization functions are now methods of `Description`. The `pyparcs.RandomDescription` presents the graph randomization.
- **Guideline objects**: `pyparcs.Guideline` and `pyparcs.GuidelineIterator` handle the functionalities about providing randomization guidelines.
- **Graph objects**: `pyparcs.Graph` object now receives description objects as input (instead of node/edge). It throws an error if the description is still partially specified.


## Improvements

- **Reading outline information**: the `Description.outline` attribute can be called at any time (before or after randomization) to return the updated outline information.
- **Randomization tags**: by assigning tags to outline lines, we can mark the nodes and edges for selective randomization. This way, a partially-specified outline can be randomized using more than one guideline.
- **Randomizing coefficients**: randomizing single coefficients (e.g. `mu_=2X-?Y`) is now possible.
- **infer edges**: trivial edges can be inferred, based on the name of the parent nodes which appear in child node lines.
- **faster sampling**: sampling procedures is in the order of x100 faster after code optimization.

## Documentation

API and diagrams are added to the documentation, for better understanding of the PARCS mechanics.