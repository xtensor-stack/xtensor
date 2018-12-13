.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Operators and related functions
===============================

Defined in ``xtensor/xmath.hpp`` and ``xtensor/xoperation.hpp``

.. _identity-op-ref:
.. doxygenfunction:: operator+(E&&)
   :project: xtensor

.. _neg-op-ref:
.. doxygenfunction:: operator-(E&&)
   :project: xtensor

.. _plus-op-ref:
.. doxygenfunction:: operator+(E1&&, E2&&)
   :project: xtensor

.. _minus-op-ref:
.. doxygenfunction:: operator-(E1&&, E2&&)
   :project: xtensor

.. _mul-op-ref:
.. doxygenfunction:: operator*(E1&&, E2&&)
   :project: xtensor

.. _div-op-ref:
.. doxygenfunction:: operator/(E1&&, E2&&)
   :project: xtensor

.. _or-op-ref:
.. doxygenfunction:: operator||(E1&&, E2&&)
   :project: xtensor

.. _and-op-ref:
.. doxygenfunction:: operator&&(E1&&, E2&&)
   :project: xtensor

.. _not-op-ref:
.. doxygenfunction:: operator!(E&&)
   :project: xtensor

.. _where-op-ref:
.. doxygenfunction:: where(E1&&, E2&&, E3&&)
   :project: xtensor

.. _any-op-ref:
.. doxygenfunction:: any(E&&)
   :project: xtensor

.. _all-op-ref:
.. doxygenfunction:: all(E&&)
   :project: xtensor

.. _less-op-ref:
.. doxygenfunction:: operator<(E1&&, E2&&)
   :project: xtensor

.. _less-eq-op-ref:
.. doxygenfunction:: operator<=(E1&&, E2&&)
   :project: xtensor

.. _greater-op-ref:
.. doxygenfunction:: operator>(E1&&, E2&&)
   :project: xtensor

.. _greater-eq-op-ref:
.. doxygenfunction:: operator>=(E1&&, E2&&)
   :project: xtensor

.. _equal-op-ref:
.. doxygenfunction:: operator==(const xexpression<E1>&, const xexpression<E2>&)
   :project: xtensor

.. _nequal-op-ref:
.. doxygenfunction:: operator!=(const xexpression<E1>&, const xexpression<E2>&)
   :project: xtensor

.. _equal-fn-ref:
.. doxygenfunction:: equal(E1&&, E2&&)
   :project: xtensor

.. _nequal-fn-ref:
.. doxygenfunction:: not_equal(E1&&, E2&&)
   :project: xtensor

.. _less-fn-ref:
.. doxygenfunction:: less(E1&& e1, E2&& e2)
   :project: xtensor

.. _less-eq-fn-ref:
.. doxygenfunction:: less_equal(E1&& e1, E2&& e2)
   :project: xtensor

.. _greater-fn-ref:
.. doxygenfunction:: greater(E1&& e1, E2&& e2)
   :project: xtensor

.. _greate-eq-fn-ref:
.. doxygenfunction:: greater_equal(E1&& e1, E2&& e2)
   :project: xtensor

.. _bitwise-and-op-ref:
.. doxygenfunction:: operator&(E1&&, E2&&)
   :project: xtensor

.. _bitwise-or-op-ref:
.. doxygenfunction:: operator|(E1&&, E2&&)
   :project: xtensor

.. _bitwise-xor-op-ref:
.. doxygenfunction:: operator^(E1&&, E2&&)
   :project: xtensor

.. _bitwise-not-op-ref:
.. doxygenfunction:: operator~(E&&)
   :project: xtensor

.. _left-shift-fn-ref:
.. doxygenfunction:: left_shift(E1&&, E2&&)
   :project: xtensor

.. _right-shift-fn-ref:
.. doxygenfunction:: right_shift(E1&&, E2&&)
   :project: xtensor

.. _left-sh-op-ref:
.. doxygenfunction:: operator<<(E1&&, E2&&)
   :project: xtensor

.. _right-sh-op-ref:
.. doxygenfunction:: operator>>(E1&&, E2&&)
   :project: xtensor

.. _cast-ref:
.. doxygenfunction:: cast(E&&)
   :project: xtensor
