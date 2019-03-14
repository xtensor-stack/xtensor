.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Mathematical functions
======================

Operations and functions of ``xtensor`` are not evaluated until they are assigned.
In the following, ``e1``, ``e2`` and ``e3`` can be arbitrary tensor expressions.
The results of operations and functions are assigned to ``xt::xarray`` in the examples,
but that could be any other container (or even views). To keep an unevaluated
operator / function, assign to an ``auto`` variable:

.. code::

    auto res = e1 + e2;

See :ref:`lazy-evaluation` for more details on unevaluated expressions.

Basic functions
---------------

.. code::

    xt::xarray<double> res0 = xt::abs(e1);
    xt::xarray<double> res1 = xt::fabs(e1);
    xt::xarray<double> res2 = xt::fmod(e1, e2);
    xt::xarray<double> res3 = xt::remainder(e1, e2);
    xt::xarray<double> res4 = xt::fma(e1, e2, e3);
    xt::xarray<double> res5 = xt::maximum(e1, e2);
    xt::xarray<double> res6 = xt::minimum(e2, e2);
    xt::xarray<double> res7 = xt::fmax(e1, e2);
    xt::xarray<double> res8 = xt::fmin(e1, e2);
    xt::xarray<double> res9 = xt::fdim(e1, e2);
    xt::xarray<double> res10 = xt::clip(e1, e2, e3);
    xt::xarray<double> res11 = xt::sign(e1);

Exponential functions
---------------------

.. code::

    xt::xarray<double> res0 = xt::exp(e1);
    xt::xarray<double> res2 = xt::exp2(e1);
    xt::xarray<double> res3 = xt::expm1(e1);
    xt::xarray<double> res4 = xt::log(e1);
    xt::xarray<double> res5 = xt::log2(e1);
    xt::xarray<double> res6 = xt::log10(e1);
    xt::xarray<double> res7 = xt::log1p(e1);

Power functions
---------------

.. code::

    xt::xarray<double> res0 = xt::pow(e1, e2);
    xt::xarray<double> res1 = xt::sqrt(e1);
    xt::xarray<double> res2 = xt::cbrt(e1);
    xt::xarray<double> res3 = xt::hypot(e1, e2);

Trigonometric functions
-----------------------

.. code::

    xt::xarray<double> res0 = xt::cos(e1);
    xt::xarray<double> res1 = xt::sin(e1);
    xt::xarray<double> res2 = xt::tan(e1);
    xt::xarray<double> res3 = xt::acos(e2);
    xt::xarray<double> res4 = xt::asin(e2);
    xt::xarray<double> res5 = xt::atan(e2);
    xt::xarray<double> res6 = xt::atan2(e2, e3);

Hyperbolic functions
--------------------

.. code::

    xt::xarray<double> res0 = xt::cosh(e1);
    xt::xarray<double> res1 = xt::sinh(e1);
    xt::xarray<double> res2 = xt::tanh(e1);
    xt::xarray<double> res3 = xt::acosh(e2);
    xt::xarray<double> res4 = xt::asinh(e2);
    xt::xarray<double> res5 = xt::atanh(e2);

Error and gamma functions
-------------------------

.. code::

    xt::xarray<double> res0 = xt::erf(e1);
    xt::xarray<double> res1 = xt::erfc(e1);
    xt::xarray<double> res2 = xt::tgamma(e1);
    xt::xarray<double> res3 = xt::lgamma(e1);

Nearest integer operations
--------------------------

.. code::

    xt::xarray<double> res0 = xt::ceil(e1);
    xt::xarray<double> res1 = xt::floor(e1);
    xt::xarray<double> res2 = xt::trunc(e1);
    xt::xarray<double> res3 = xt::round(e1);
    xt::xarray<double> res4 = xt::nearbyint(e1);
    xt::xarray<double> res5 = xt::rint(e1);

Classification functions
------------------------

.. code::

    xt::xarray<double> res0 = xt::isfinite(e1);
    xt::xarray<double> res1 = xt::isinf(e1);
    xt::xarray<double> res2 = xt::isnan(e1);
    xt::xarray<double> res3 = xt::isclose(e1, e2);
    bool res4 = xt::allclose(e1, e2);

