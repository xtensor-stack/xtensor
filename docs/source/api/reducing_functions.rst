.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Reducing functions
==================

**xtensor** provides the following reducing functions for xexpressions:

Defined in ``xtensor/xmath.hpp``

.. _sum-function-reference:
.. doxygenfunction:: sum(E&&, X&&, ES)
   :project: xtensor

.. _prod-function-reference:
.. doxygenfunction:: prod(E&&, X&&, ES)
   :project: xtensor

.. _mean-function-reference:
.. doxygenfunction:: mean(E&&, X&&)
   :project: xtensor

Defined in ``xtensor/xnorm.hpp``

.. _norm-l0-func-ref:
.. doxygenfunction:: norm_l0(E&&, X&&)
   :project: xtensor

.. _norm-l1-func-ref:
.. doxygenfunction:: norm_l1(E&&, X&&)
   :project: xtensor

.. _norm-sq-func-ref:
.. doxygenfunction:: norm_sq(E&&, X&&)
   :project: xtensor

.. _norm-l2-func-ref:
.. doxygenfunction:: norm_l2(E&&, X&&)
   :project: xtensor

.. _norm-linf-func-ref:
.. doxygenfunction:: norm_linf(E&&, X&&)
   :project: xtensor

.. _nlptop-func-ref:
.. doxygenfunction:: norm_lp_to_p(E&&, double, X&&)
   :project: xtensor

.. _norm-lp-func-ref:
.. doxygenfunction:: norm_lp(E&&, double, X&&)
   :project: xtensor

.. _nind-l1-ref:
.. doxygenfunction:: norm_induced_l1(E&&)
   :project: xtensor

.. _nilinf-ref:
.. doxygenfunction:: norm_induced_linf(E&&)
   :project: xtensor
