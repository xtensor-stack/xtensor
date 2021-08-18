.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

NaN functions
=============

**xtensor** provides the following functions that deal with NaNs in xexpressions:

Defined in ``xtensor/xmath.hpp``

.. doxygenfunction:: nan_to_num(E&&)
   :project: xtensor

.. doxygenfunction:: nanmin(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nanmax(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nansum(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nanmean(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nanvar(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nanstd(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nanprod(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: nancumsum(E&&)
   :project: xtensor

.. doxygenfunction:: nancumsum(E&&, std::ptrdiff_t)
   :project: xtensor

.. doxygenfunction:: nancumprod(E&&)
   :project: xtensor

.. doxygenfunction:: nancumprod(E&&, std::ptrdiff_t)
   :project: xtensor
