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
from typing import Union, Iterator
from typeguard import typechecked
from itertools import product
import numpy as np
from copy import deepcopy
from benedict import benedict
from pyparcs.api.utils import digest_outline_input
from pyparcs.api.randomization import *
from pyparcs.core.exceptions import parcs_assert, GuidelineError


@typechecked
class Guideline:
    """guideline objects for PARCS randomization

    Guideline object stores, parses and controls the user-given
    guideline outline for randomization. It will then be passed
    to the randomization methods of the Descriptions,
    or to the RandomDescription class.

    Parameters
    ----------
    outline : dict or path
        a dictionary or a yml file provided by a path which provides
        the guideline directives, etc.

    """

    def __init__(self, outline: Union[dict, str]):
        # == USER INPUTS ==
        # store the dict format of the given outlines
        self.outline = digest_outline_input(outline)
        self.guide = benedict(outline)

    @staticmethod
    def _directive_sampler(directive: Union[list, int, float]):
        """samples from directives

        directives are in one of the following formats:

        1. ["choice", "A", "B", "C", ...]
        2-3. ["f-range" OR "i-range", 0, 8, 10, 12, ...]
        4. Single values e.g. 1

        for 2-3, pairs of values specify ranges

        Returns
        -------
        sampled value
        """
        if isinstance(directive, list):
            if directive[0] == 'choice':
                options = directive[1:]
                return np.random.choice(options)
            else:
                ranges = directive[1:]
                # if multiple ranges are given
                parcs_assert(len(ranges) % 2 == 0,
                             GuidelineError,
                             f'The number of range bounds should be even, got {len(ranges)}')
                num_ranges = int(len(ranges) / 2)
                # first, pick a random range
                range_ = np.random.randint(num_ranges)
                # then, pick from within the selected range
                low, high = ranges[range_ * 2], ranges[range_ * 2 + 1]
                if directive[0] == 'i-range':
                    return np.random.randint(low=low, high=high)
                if directive[0] == 'f-range':
                    return np.random.uniform(low=low, high=high)
                raise ValueError
        else:
            # it is a fixed value, hence no sampling
            return directive

    def sample_keys(self, which: str):
        """sample the node distribution or edge functions

        Sampling is done from a uniform discrete over options
        using numpy.random.choice

        Parameters
        ----------
        which: ['nodes', 'edges']
            specify if sampling from nodes or edges

        Returns
        -------
        sampled_key
        """
        return np.random.choice(list(self.outline[which].keys()))

    def sample_values(self, keypath: str):
        """Sampling from values

        Sampling from uniform discrete or continuous, from the values
        of the guideline outline.

        Parameters
        ----------
        keypath: str
            path to the value, used by python-benedict dict

        Returns
        -------
        sampled_value
        """
        try:
            ind = int((split_keypath := keypath.split('.'))[-1])
            keypath = '.'.join(split_keypath[:-1])
            return self._directive_sampler(self.guide[keypath][ind])
        except ValueError:
            # last one is not an index:
            return self._directive_sampler(self.guide[keypath])


class GuidelineIterator:
    """Generating guidelines based on a directive

    By passing a standard guideline outline to this class, it
    provides methods to return equivalent guidelines that
    differ only in one of the targeted directives.

    It is a useful tool for analysis of the sensitivity
    of a model to a specific simulation parameter.

    Parameters
    ----------
    outline: dict or path
        a standard guideline outline
    """
    def __init__(self, outline):
        self.outline = benedict(outline)

    @staticmethod
    def _parse_route(route):
        return route.replace('bias', '0').replace('linear', '1').replace('interactions', '2')

    def get_guidelines(self, route: str,
                       steps: Union[int, float, None] = None) -> Iterator[dict]:
        """Get the guideline generator

        The output is a generator that we can use in a for loop to
        loop over possible values in the guideline

        Parameters
        ----------
        route: str
            serialized key route to the value
        steps: int or float or None
            steps to iterate over the directive. Must be None for
            choice, and int for i-range

        Returns
        -------
        generator
        """
        route = self._parse_route(route)

        try:
            ind = int((split_keypath := route.split('.'))[-1])
            keypath = '.'.join(split_keypath[:-1])
            directive = self.outline[keypath][ind]
        except ValueError:
            # last one is not an index:
            directive = self.outline[route]

        parcs_assert(isinstance(directive, list),
                     GuidelineError,
                     f'invalid directive {directive}. iteration is possible over list directives.')
        parcs_assert(directive[0] in ['f-range', 'i-range', 'choice'],
                     GuidelineError,
                     f'invalid directive type {directive[0]}')
        parcs_assert(len(directive) == 3 or directive[0] == 'choice',
                     GuidelineError,
                     'guideline iterator does not support multi range directives')
        parcs_assert(directive[0] != 'choice' or steps is None,
                     GuidelineError,
                     'iterator `steps` not defined for a `choice` directive type.')
        parcs_assert(directive[0] in ['choice', 'f-range'] or isinstance(steps, int),
                     GuidelineError,
                     'steps must be int for an `i-range` directive type.')

        range_ = (directive[1:] if directive[0] == 'choice' else
                  np.arange(directive[1], directive[2], steps) if directive[0] == 'f-range'
                  else range(directive[1], directive[2], steps))

        for value in range_:
            frozen_outline = benedict(deepcopy(self.outline))
            try:
                ind = int((split_keypath := route.split('.'))[-1])
                keypath = '.'.join(split_keypath[:-1])
                frozen_outline[keypath][ind] = value
            except ValueError:
                # last one is not an index:
                frozen_outline[route] = value
            yield Guideline(frozen_outline)

