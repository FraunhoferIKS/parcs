#  Copyright (c) 2022. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
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
import inspect
import numpy as np


class DistributionError(Exception):
    """
    Exception class for issues related to the Node output distribution
    """
    def __init__(self, msg):
        super().__init__(msg)


class EdgeFunctionError(Exception):
    """
    Exception class for issues related to the edge functions
    """
    def __init__(self, msg):
        super().__init__(msg)


class GraphError(Exception):
    """
    Exception class for issues related to the graph object
    """
    def __init__(self, msg):
        super().__init__(msg)


class GuidelineError(Exception):
    """
    Exception class for issues related to the randomization guideline file
    """
    def __init__(self, msg):
        super().__init__(msg)


class DescriptionFileError(Exception):
    """
    Exception class for issues related to the graph description file
    """
    def __init__(self, msg):
        super().__init__(msg)


class ExternalResourceError(Exception):
    """
    Exception class for issues related to the external resources e.g. CSV files
    """
    def __init__(self, msg):
        super().__init__(msg)


class RandomizerError(Exception):
    """
    Exception class for issues related to the randomizer objects
    """
    def __init__(self, msg):
        super().__init__(msg)


class NodeError(Exception):
    """
    Exception class for issues related to the node objects
    """
    def __init__(self, msg):
        super().__init__(msg)


def parcs_assert(condition, action, msg):
    """**PARCS-specific assertions**

    This functions helps with asserting conditions and raising proper error and msg.

    Parameters
    ----------
    condition : bool
        Condition to be checked. If `False` then raises the error
    action : Exception
        The exception to be raised if condition is False
    msg : str
        The exception message that is shown

    Returns
    -------

    """
    if not condition:
        raise action(msg)


def validate_error_term(arr, node_name):
    try:
        parcs_assert(
            np.all(0 <= arr) and np.all(1 > arr),
            ValueError,
            f"Error for Node {node_name}: Error term must be in the range [0,1)"
        )
    except Exception as e:
        raise ValueError(f"Error for Node {node_name}") from e


def validate_deterministic_function(func, node_name):
    sig = inspect.signature(func)
    parcs_assert(
        len(sig.parameters) == 1,
        ExternalResourceError,
        (f"Error for Node {node_name}:\n"
         f"Deterministic function {func.__name__} has more than 1 input parameter. "
         "To process parent nodes, you must assume the input is a pandas DataFrame, "
         "and treat parents as columns of the DataFrame. \n"
         "You should not assume parents to be given to the function separately."
         "Example: lambda data: data['Z_1'] + data['Z_2']")
    )
    param_type = next(iter(sig.parameters.items()))[1].kind
    parcs_assert(
        param_type == inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ExternalResourceError,
        (f"Error for Node {node_name}:\n"
         f"The parameter of function {func.__name__} is not positional (or keyword)")
    )