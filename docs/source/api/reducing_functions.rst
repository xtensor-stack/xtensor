.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Reducing functions
==================

**xtensor** provides the following reducing functions for xexpressions:

Defined in ``xtensor/xmath.hpp``

.. doxygenfunction:: sum(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: sum(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: prod(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: prod(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: mean(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: mean(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: average(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: variance(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: variance(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: variance(E&&, X&&, const D&, EVS)
   :project: xtensor

.. doxygenfunction:: stddev(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: stddev(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: diff(const xexpression<T>&, unsigned int, std::ptrdiff_t)
   :project: xtensor

.. doxygenfunction:: amax(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: amax(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: amin(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: amin(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: trapz(const xexpression<T>&, double, std::ptrdiff_t)
   :project: xtensor

.. doxygenfunction:: trapz(const xexpression<T>&, const xexpression<E>&, std::ptrdiff_t)
   :project: xtensor

Defined in ``xtensor/xnorm.hpp``

.. doxygenfunction:: norm_l0(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_l1(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_sq(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_l2(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_linf(E&&, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_lp_to_p(E&&, double, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_lp(E&&, double, X&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_induced_l1(E&&, EVS)
   :project: xtensor

.. doxygenfunction:: norm_induced_linf(E&&, EVS)
   :project: xtensor
