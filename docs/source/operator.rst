.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Operators and functions
=======================

Arithmetic operators
--------------------

`xtensor` provides overloads of traditional arithmetic operators for ``xexpression`` objects:

- unary ``operator+``
- unary ``operator-``
- ``operator+``
- ``operator-``
- ``operator*``
- ``operator/``

All these operators are element-wise operators and apply the lazy broadcasting rules explained in
a previous section.

.. code::

    #incude "xtensor/xarray.hpp"

    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> b = {1, 2};

    xt::xarray<int> res = 2 * (a + b);
    // => res = {{4, 8}, {8, 12}}

Logical operators
-----------------

`xtensor` also provides overloads of the logical operators:

- ``operator!``
- ``operator||``
- ``operator&&``

Like arithmetic operators, these logical operators are element-wise operators and apply the lazy broadcasting
rules. In addition to these element-wise logical operators, `xtensor` provides two reducing boolean functions:

- ``any(E&& e)`` returns ``true`` if any of ``e`` elements is truthy, ``false`` otherwise.
- ``all(E&& e)`` returns ``true`` if all alements of ``e`` are truthy, ``false`` otherwise.

and an element-wise ternary function (similar to the ``: ?`` ternary operator):

- ``where(E&& b, E1&& e&, E2&& e2)`` returns an ``xexpression`` whose elements are those of ``e1``
  when corresponding elements of ``b`` are thruthy, and those of ``e2`` otherwise.

.. code::

    #include "xtensor/xarray.hpp"

    xt::array<bool> b = { false, true, true, false };
    xt::array<int> a1 = { 1,   2,  3,  4 };
    xt::array<int> a2 = { 11, 12, 13, 14 };

    xt::array<int> res = xt::where(b, a1, a2);
    // => res = { 11, 2, 3, 14 }

Unlike in ``numpy.where``, ``xt::where`` takes full advantage of the lazyness of `xtensor`.

Comparison operators
--------------------

`xtensor` provides overloads of the inequality operators:

- ``operator<``
- ``operator<=``
- ``operator>``
- ``operator>=``

These overloads of inequality operators are quite different from the standard C++ inequality operators: they are element-wise
operators returning boolean ``xexpression``:

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<int> a1 = {  1, 12,  3, 14 };
    xt::xarray<int> a2 = { 11,  2, 13, 4  };
    xt::xarray<bool> comp = a1 < a2;
    // => comp = { true, false, true, false }

However, equality operators are similar to the traditional ones in C++:

- ``operator==(const E1& e1, const E2& e2)`` returns ``true`` if ``e1`` and ``e2`` hold the same elements.
- ``operator!=(const E1& e1, const E2& e2)`` returns ``true`` if ``e1`` and ``e2`` don't hold the same elements.

Element-wise equality comparison can be achieved through the ``xt::equal`` function.

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<int> a1 = {  1,  2, 3, 4};
    xt::xarray<int> a2 = { 11, 12, 3, 4};

    bool res = (a1 == a2);
    // => res = false

    xt::xarray<bool> re = xt::equal(a1, a2);
    // => re = { false, false, true, true }

Mathematical functions
----------------------
`xtensor` provides overloads for many of the standard mathematical functions:

- basic functions: ``abs``, ``remainder``, ``fma``, ...
- exponential functions: ``exp``, ``expm1``, ``log``, ``log1p``, ...
- power functions: ``pow``, ``sqrt``, ``cbrt``, ...
- trigonometric functions: ``sin``, ``cos``, ``tan``, ...
- hyperbolic functions: ``sinh``, ``cosh``, ``tanh``, ...
- Error and gamma functions: ``erf``, ``erfc``, ``tgamma``, ``lgamma``, ....

See the API reference for a comprehensive list of available functions. Like operators, the mathematical functions
are element-wise functions and apply the lazy broadcasting rules.

Universal functions and vectorization
-------------------------------------

`xtensor` provides utilities to **vectorize any scalar function** (taking multiple scalar arguments) into a function that
will perform on ``xexpression`` s, applying the lazy broadcasting rules which we described in a previous section. These
functions are called ``xfunction`` s. They are `xtensor`'s counterpart to numpy's universal functions.

Actually, all arithmetic and logical operators, inequality operator and mathematical functions we described before are
``xfunction`` s.

The following snippet shows how to vectorize a scalar function taking two arguments:

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xvectorize.hpp"

    int f(int a, int b)
    {
        return a + 2 * b;
    }

    auto vecf = xt::vectorize(f);
    xt::array<int> a = { 11, 12, 13 };
    xt::array<int> b = {  1,  2,  3 };
    xt::array<int> res = vecf(a, b);
    // => res = { 13, 16, 19 }

