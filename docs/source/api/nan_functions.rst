.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

NaN functions
=============

**xtensor** provides the following functions that deal with NaNs in xexpressions:

Defined in ``xtensor/xmath.hpp``

.. doxygenfunction:: nan_to_num(E&&)

.. doxygenfunction:: nanmin(E&&, X&&, EVS)

.. doxygenfunction:: nanmax(E&&, X&&, EVS)

.. doxygenfunction:: nansum(E&&, X&&, EVS)

.. doxygenfunction:: nanmean(E&&, X&&, EVS)

.. doxygenfunction:: nanvar(E&&, X&&, EVS)

.. doxygenfunction:: nanstd(E&&, X&&, EVS)

.. doxygenfunction:: nanprod(E&&, X&&, EVS)

.. doxygenfunction:: nancumsum(E&&)

.. doxygenfunction:: nancumsum(E&&, std::ptrdiff_t)

.. doxygenfunction:: nancumprod(E&&)

.. doxygenfunction:: nancumprod(E&&, std::ptrdiff_t)
