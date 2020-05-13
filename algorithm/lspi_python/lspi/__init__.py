# -*- coding: utf-8 -*-
"""Least Squares Policy Iteration (LSPI) implementation.

Implements the algorithms described in the paper

"Least-Squares Policy Iteration."
Lagoudakis, Michail G., and Ronald Parr.
Journal of Machine Learning Research 4, 2003.
https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf

The implementation is based on the Matlab implementation provided by
the authors. The implementation is available for download at
http://www.cs.duke.edu/research/AI/LSPI/

"""

import algorithm.lspi_python.lspi.basis_functions  # noqa
import algorithm.lspi_python.lspi.domains  # noqa
from algorithm.lspi_python.lspi.lspi import learn  # noqa
from algorithm.lspi_python.lspi.policy import Policy  # noqa
from algorithm.lspi_python.lspi.sample import Sample  # noqa
import algorithm.lspi_python.lspi.solvers  # noqa
