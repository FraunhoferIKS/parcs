## Features

- **Temporal Causal DAGs**: Defined by their unique description file (via `pyparcs.graph_builder.temporal_parsers`)
- **Visualization**: Via the _Graphviz_ library, the `Graph.visualize()` method returns a static visualization.

## Improvements

- **Custom path for determinist nodes**: The custom Python file for determinist nodes doesn't need to be in the same folder as the main script
- **description dictionaries**: We can pass the descriptions as dictionaries rather than yml files
- **No unnecessary burn-out**: Graphs without correction elements won't run a burn-out phase
- **Informative error message in description**: wrong parameterizing of the distributions now throws an informative error message

## Bug fixes

- **Invalid parents for `.do()`** Now choosing the descendant nodes of a node for functional intervention throws an informative error