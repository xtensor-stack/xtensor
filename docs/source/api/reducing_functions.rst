.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Reducing functions
==================

**xtensor** provides the following reducing functions for xexpressions:

Defined in ``xtensor/xmath.hpp``

.. _sum-function-reference:
.. doxygenfunction:: sum(E&&, X&&, EVS)
   :project: xtensor

.. _prod-function-reference:
.. doxygenfunction:: prod(E&&, X&&, EVS)
   :project: xtensor

.. _mean-function-reference:
.. doxygenfunction:: mean(E&&, X&&, EVS)
   :project: xtensor

.. _variance-function-reference:
.. doxygenfunction:: variance(E&&, X&&, EVS)
   :project: xtensor

.. _stddev-function-reference:
.. doxygenfunction:: stddev(E&&, X&&, EVS)
   :project: xtensor

.. _diff-function-reference:
.. doxygenfunction:: diff(const xexpression<T>&, unsigned int, std::ptrdiff_t)
   :project: xtensor

.. _amax-function-reference:
.. doxygenfunction:: amax(E&&, X&&, EVS)
   :project: xtensor

.. _amin-function-reference:
.. doxygenfunction:: amin(E&&, X&&, EVS)
   :project: xtensor

.. _trapz-function-reference:
.. doxygenfunction:: trapz(const xexpression<T>&, double, std::ptrdiff_t)
   :project: xtensor

.. _trapz-function-reference2:
.. doxygenfunction:: trapz(const xexpression<T>&, const xexpression<E>&, std::ptrdiff_t)
   :project: xtensor

Defined in ``xtensor/xnorm.hpp``

.. _norm-l0-func-ref:
.. doxygenfunction:: norm_l0(E&&, X&&, EVS)
   :project: xtensor

.. _norm-l1-func-ref:
.. doxygenfunction:: norm_l1(E&&, X&&, EVS)
   :project: xtensor

.. _norm-sq-func-ref:
.. doxygenfunction:: norm_sq(E&&, X&&, EVS)
   :project: xtensor

.. _norm-l2-func-ref:
.. doxygenfunction:: norm_l2(E&&, X&&, EVS)
   :project: xtensor

.. _norm-linf-func-ref:
.. doxygenfunction:: norm_linf(E&&, X&&, EVS)
   :project: xtensor

.. _nlptop-func-ref:
.. doxygenfunction:: norm_lp_to_p(E&&, double, X&&, EVS)
   :project: xtensor

.. _norm-lp-func-ref:
.. doxygenfunction:: norm_lp(E&&, double, X&&, EVS)
   :project: xtensor

.. _nind-l1-ref:
.. doxygenfunction:: norm_induced_l1(E&&, EVS)
   :project: xtensor

.. _nilinf-ref:
.. doxygenfunction:: norm_induced_linf(E&&, EVS)
   :project: xtensor
