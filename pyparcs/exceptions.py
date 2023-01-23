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
import numpy as np
import inspect


class DistributionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class EdgeFunctionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class GraphError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class GuidelineError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class DescriptionFileError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class ExternalResourceError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class RandomizerError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class NodeError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def parcs_assert(condition, action, msg):
    if not condition:
        raise action(msg)


def validate_error_term(arr, node_name):
    try:
        parcs_assert(
            np.all(0 <= arr) and np.all(1 > arr),
            ValueError,
            '''Error for Node {}:
            Error term must be in the range [0,1)'''.format(node_name)
        )
    except Exception as e:
        raise ValueError('''Error for Node {}:
        Error while reading error terms: {}'''.format(node_name, e))


def validate_deterministic_function(func, node_name):
    sig = inspect.signature(func)
    parcs_assert(
        len(sig.parameters) == 1,
        ExternalResourceError,
        '''Error for Node {}:
        Deterministic function {} has more than 1 input parameter.
        To process parent nodes, you must assume the input is a pandas DataFrame, and treat parents as columns
        of the DataFrame. You should not assume parents to be given to the function separately.
        Example: lambda data: data['Z_1'] + data['Z_2']
        '''.format(node_name, func.__name__)
    )
    param_type = next(iter(sig.parameters.items()))[1].kind
    parcs_assert(
        param_type == inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ExternalResourceError,
        '''Error for Node {}:
        The parameter of function {} is not positional (or keyword)'''.format(node_name, func.__name__)
    )