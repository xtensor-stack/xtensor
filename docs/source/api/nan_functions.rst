.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

NaN functions
=============

**xtensor** provides the following functions that deal with NaNs in xexpressions:

Defined in ``xtensor/xmath.hpp``

.. _nan-to-num-function-reference:
.. doxygenfunction:: nan_to_num(E&&)
   :project: xtensor

.. _nansum-function-reference:
.. doxygenfunction:: nansum(E&&, X&&, EVS)
   :project: xtensor

.. _nanprod-function-reference:
.. doxygenfunction:: nanprod(E&&, X&&, EVS)
   :project: xtensor

.. _nancumsum-function-reference:
.. doxygenfunction:: nancumsum(E&&)
   :project: xtensor

.. doxygenfunction:: nancumsum(E&&, std::ptrdiff_t)
   :project: xtensor

.. _nancumprod-function-reference:
.. doxygenfunction:: nancumprod(E&&)
   :project: xtensor

.. doxygenfunction:: nancumprod(E&&, std::ptrdiff_t)
   :project: xtensor
