
.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Mathematical functions
======================

Operations and functions of ``xtensor`` are not evaluated until they are assigned.
In the following, ``e1``, ``e2`` and ``e3`` can be arbitrary tensor expressions.

Basic functions
---------------

.. code::

    auto res0 = xt::abs(e1);
    auto res1 = xt::fabs(e1);
    auto res2 = xt::fmod(e1, e2);
    auto res3 = xt::remainder(e1, e2);
    auto res4 = xt::fma(e1, e2, e3);
    auto res5 = xt::maximum(e1, e2);
    auto res6 = xt::minimum(e2, e2);
    auto res7 = xt::fmax(e1, e2);
    auto res8 = xt::fmin(e1, e2);
    auto res9 = xt::fdim(e1, e2);
    auto res10 = xt::clip(e1, e2, e3);
    auto res11 = xt::sign(e1);

Exponential functions
---------------------

.. code::

    auto res0 = xt::exp(e1);
    auto res2 = xt::exp2(e1);
    auto res3 = xt::expm1(e1);
    auto res4 = xt::log(e1);
    auto res5 = xt::log2(e1);
    auto res6 = xt::log10(e1);
    auto res7 = xt::log1p(e1);

Power functions
---------------

.. code::

    auto res0 = xt::pow(e1, e2);
    auto res1 = xt::sqrt(e1);
    auto res2 = xt::cbrt(e1);
    auto res3 = xt::hypot(e1, e2);

Trigonometric functions
-----------------------

.. code::

    auto res0 = xt::cos(e1);
    auto res1 = xt::sin(e1);
    auto res2 = xt::tan(e1);
    auto res3 = xt::acos(e2);
    auto res4 = xt::asin(e2);
    auto res5 = xt::atan(e2);
    auto res6 = xt::atan2(e2, e3);

Hyperbolic functions
--------------------

.. code::

    auto res0 = xt::cosh(e1);
    auto res1 = xt::sinh(e1);
    auto res2 = xt::tanh(e1);
    auto res3 = xt::acosh(e2);
    auto res4 = xt::asinh(e2);
    auto res5 = xt::atanh(e2);

Error and gamma functions
-------------------------

.. code::

    auto res0 = xt::erf(e1);
    auto res1 = xt::erfc(e1);
    auto res2 = xt::tgamma(e1);
    auto res3 = xt::lgamma(e1);

Nearest integer operations
--------------------------

.. code::

    auto res0 = xt::ceil(e1);
    auto res1 = xt::floor(e1);
    auto res2 = xt::trunc(e1);
    auto res3 = xt::round(e1);
    auto res4 = xt::nearbyint(e1);
    auto res5 = xt::rint(e1);

Classification functions
------------------------

.. code::

    auto res0 = xt::isfinite(e1);
    auto res1 = xt::isinf(e1);
    auto res2 = xt::isnan(e1);
    auto res3 = xt::isclose(e1, e2);
    bool res4 = xt::allclose(e1, e2);
