.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Reducing functions
==================

**xtensor** provides the following reducing functions for xexpressions:

Defined in ``xtensor/xmath.hpp``

.. doxygenfunction:: sum(E&&, EVS)

.. doxygenfunction:: sum(E&&, X&&, EVS)

.. doxygenfunction:: prod(E&&, EVS)

.. doxygenfunction:: prod(E&&, X&&, EVS)

.. doxygenfunction:: mean(E&&, EVS)

.. doxygenfunction:: mean(E&&, X&&, EVS)

.. doxygenfunction:: average(E&&, EVS)

.. doxygenfunction:: variance(E&&, EVS)

.. doxygenfunction:: variance(E&&, X&&, EVS)

.. doxygenfunction:: variance(E&&, X&&, const D&, EVS)

.. doxygenfunction:: stddev(E&&, EVS)

.. doxygenfunction:: stddev(E&&, X&&, EVS)

.. doxygenfunction:: diff(const xexpression<T>&, unsigned int, std::ptrdiff_t)

.. doxygenfunction:: amax(E&&, EVS)

.. doxygenfunction:: amax(E&&, X&&, EVS)

.. doxygenfunction:: amin(E&&, EVS)

.. doxygenfunction:: amin(E&&, X&&, EVS)

.. doxygenfunction:: trapz(const xexpression<T>&, double, std::ptrdiff_t)

.. doxygenfunction:: trapz(const xexpression<T>&, const xexpression<E>&, std::ptrdiff_t)

Defined in ``xtensor/xnorm.hpp``

.. doxygenfunction:: norm_l0(E&&, X&&, EVS)

.. doxygenfunction:: norm_l1(E&&, X&&, EVS)

.. doxygenfunction:: norm_sq(E&&, X&&, EVS)

.. doxygenfunction:: norm_l2(E&&, X&&, EVS)

.. doxygenfunction:: norm_linf(E&&, X&&, EVS)

.. doxygenfunction:: norm_lp_to_p(E&&, double, X&&, EVS)

.. doxygenfunction:: norm_lp(E&&, double, X&&, EVS)

.. doxygenfunction:: norm_induced_l1(E&&, EVS)

.. doxygenfunction:: norm_induced_linf(E&&, EVS)
