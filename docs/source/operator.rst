.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Operators and functions
=======================

Arithmetic operators
--------------------

*xtensor* provides overloads of traditional arithmetic operators for
:cpp:type:`xt::xexpression` objects:

- unary :cpp:func:`~xt::xexpression::operator+`
- unary :cpp:func:`~xt::xexpression::operator-`
- :cpp:func:`~xt::xexpression::operator+`
- :cpp:func:`~xt::xexpression::operator-`
- :cpp:func:`~xt::xexpression::operator*`
- :cpp:func:`~xt::xexpression::operator/`
- :cpp:func:`~xt::xexpression::operator%`

All these operators are element-wise operators and apply the lazy broadcasting
rules explained in a previous section.

.. code::

    #incude "xtensor/xarray.hpp"

    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> b = {1, 2};

    xt::xarray<int> res = 2 * (a + b);
    // => res = {{4, 8}, {8, 12}}

Logical operators
-----------------

*xtensor* also provides overloads of the logical operators:

- :cpp:func:`~xt::xexpression::operator!`
- :cpp:func:`~xt::xexpression::operator||`
- :cpp:func:`~xt::xexpression::operator&&`

Like arithmetic operators, these logical operators are element-wise operators
and apply the lazy broadcasting rules. In addition to these element-wise
logical operators, *xtensor* provides two reducing boolean functions:

- :cpp:func:`xt::any(E&& e) <xt::any>` returns ``true`` if any of ``e`` elements is truthy, ``false`` otherwise.
- :cpp:func:`xt::all(E&& e) <xt::all>` returns ``true`` if all elements of ``e`` are truthy, ``false`` otherwise.

and an element-wise ternary function (similar to the ``: ?`` ternary operator):

- :cpp:func:`xt::where(E&& b, E1&& e1, E2&& e2) <xt::where>` returns an :cpp:type:`xt::xexpression` whose elements
  are those of ``e1`` when corresponding elements of ``b`` are truthy, and
  those of ``e2`` otherwise.

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<bool> b = { false, true, true, false };
    xt::xarray<int> a1 = { 1,   2,  3,  4 };
    xt::xarray<int> a2 = { 11, 12, 13, 14 };

    xt::xarray<int> res = xt::where(b, a1, a2);
    // => res = { 11, 2, 3, 14 }

Unlike in :any:`numpy.where`, :cpp:func:`xt::where` takes full advantage of the lazyness
of *xtensor*.

Comparison operators
--------------------

*xtensor* provides overloads of the inequality operators:

- :cpp:func:`~xt::xexpression::operator\<`
- :cpp:func:`~xt::xexpression::operator\<=`
- :cpp:func:`~xt::xexpression::operator\>`
- :cpp:func:`~xt::xexpression::operator\>=`

These overloads of inequality operators are quite different from the standard
C++ inequality operators: they are element-wise operators returning boolean
:cpp:type:`xexpression`:

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<int> a1 = {  1, 12,  3, 14 };
    xt::xarray<int> a2 = { 11,  2, 13, 4  };
    xt::xarray<bool> comp = a1 < a2;
    // => comp = { true, false, true, false }

However, equality operators are similar to the traditional ones in C++:

- :cpp:func:`operator==(const E1& e1, const E2& e2) <xt::xexpression::operator==>` returns ``true`` if ``e1``
  and ``e2`` hold the same elements.
- :cpp:func:`operator!=(const E1& e1, const E2& e2) <xt::xexpression::operator!=>` returns ``true`` if ``e1``
  and ``e2`` don't hold the same elements.

Element-wise equality comparison can be achieved through the :cpp:func:`xt::equal`
function.

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<int> a1 = {  1,  2, 3, 4};
    xt::xarray<int> a2 = { 11, 12, 3, 4};

    bool res = (a1 == a2);
    // => res = false

    xt::xarray<bool> re = xt::equal(a1, a2);
    // => re = { false, false, true, true }

Bitwise operators
-----------------

*xtensor* also contains the following bitwise operators:

- Bitwise and: :cpp:func:`~xt::xexpression::operator&`
- Bitwise or: :cpp:func:`~xt::xexpression::operator|`
- Bitwise xor: :cpp:func:`~xt::xexpression::operator^`
- Bitwise not: :cpp:func:`~xt::xexpression::operator~`
- Bitwise left/right shift: :cpp:func:`~xt::xexpression::left_shift`, :cpp:func:`~xt::xexpression::right_shift`

Mathematical functions
----------------------

*xtensor* provides overloads for many of the standard mathematical functions:

- basic functions: :cpp:func:`xt::abs`, :cpp:func:`xt::remainder`, :cpp:func:`xt::fma`, ...
- exponential functions: :cpp:func:`xt::exp`, :cpp:func:`xt::expm1`, :cpp:func:`xt::log`, :cpp:func:`xt::log1p`, ...
- power functions: :cpp:func:`xt::pow`, :cpp:func:`xt::sqrt`, :cpp:func:`xt::cbrt`, ...
- trigonometric functions: :cpp:func:`xt::sin`, :cpp:func:`xt::cos`, :cpp:func:`xt::tan`, ...
- hyperbolic functions: :cpp:func:`xt::sinh`, :cpp:func:`xt::cosh`, :cpp:func:`xt::tanh`, ...
- Error and gamma functions: :cpp:func:`xt::erf`, :cpp:func:`xt::erfc`, :cpp:func:`xt::tgamma`, :cpp:func:`xt::lgamma`, ....
- Nearest integer floating point operations: :cpp:func:`xt::ceil`, :cpp:func:`xt::floor`, :cpp:func:`xt::trunc`, ...

See the API reference for a comprehensive list of available functions. Like
operators, the mathematical functions are element-wise functions and apply the
lazy broadcasting rules.

Casting
-------

*xtensor* will implicitly promote and/or cast tensor expression elements as
needed, which suffices for most use-cases. But explicit casting can be
performed via :cpp:func:`xt::cast`, which performs an element-wise ``static_cast``.

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<int> a = { 3, 5, 7 };

    auto res = a / 2;
    // => res = { 1, 2, 3 }

    auto res2 = xt::cast<double>(a) / 2;
    // => res2 = { 1.5, 2.5, 3.5 }

Reducers
--------

*xtensor* provides reducers, that is, means for accumulating values of tensor
expressions over prescribed axes. The return value of a reducer is an
:cpp:type:`xt::xexpression` with the same shape as the input expression, with the specified
axes removed.

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xmath.hpp>

    xt::xarray<double> a = xt::ones<double>({3, 2, 4, 6, 5});
    xt::xarray<double> res = xt::sum(a, {1, 3});
    // => res.shape() = { 3, 4, 5 };
    // => res(0, 0, 0) = 12

You can also call the :cpp:func:`xt::reduce` generator with your own reducing function:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    xt::xarray<double> arr = some_init_function({3, 2, 4, 6, 5});
    xt::xarray<double> res = xt::reduce([](double a, double b) { return a*a + b*b; },
                                        arr,
                                        {1, 3});

The reduce generator also accepts a :cpp:type:`xt::xreducer_functors` object, a tuple of three functions
(one for reducing, one for initialization and one for merging).
A generator is provided to build the :cpp:type:`xt::xreducer_functors` object, the last function can be omitted:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    xt::xarray<double> arr = some_init_function({3, 2, 4, 6, 5});
    xt::xarray<double> res = xt::reduce(xt::make_xreducer_functor([](double a, double b) { return a*a + b*b; },
                                                                  [](double a) { return a * 2; })
                                        arr,
                                        {1, 3});

If no axes are provided, the reduction is performed over all the axes, and the result is a 0-D expression.
Since *xtensor*'s expressions are lazy evaluated, you need to explicitely call the access operator to trigger
the evaluation and get the result:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    xt::xarray<double> arr = some_init_function({3, 2, 4, 6, 5});
    double res = xt::reduce([](double a, double b) { return a*a + b*b; }, arr)();

The ``value_type`` of a reducer is the traditional result type of the reducing operation.
For instance, the ``value_type`` of the reducer for the sum is:

- ``int`` if the underlying expression holds ``int`` values
- ``int`` if the underlying expression holds ``short`` values, because ``short + short`` = ``int``

You can pass a template argument to the reducer functions to specify the type of the initial value of
the reduction. This allows you to "promote" the value type of the reducer and limit overflows in
computation:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    xt::xarray<int> arr = some_init_function({3, 2, 4, 6, 5});
    auto s1 = xt::sum<short>(arr); // No effect, short + int = int
    auto s2 = xt::sum<long int>(arr); // The value_type of s2 is long int

When you write generic code and you want to limit overflows, you can use :cpp:any:`xt::big_promote_value_type_t`
as shown below:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    template <class E>
    void my_computation(E&& e)
    {
        auto s = xt::sum<xt::big_promote_value_type_t<E>>(e);
    }

Accumulators
------------

Similar to reducers, *xtensor* provides accumulators which are used to
implement cumulative functions such as :cpp:func:`xt::cumsum` or :cpp:func:`xt::cumprod`. Accumulators
can currently only work on a single axis. Additionally, the accumulators are
not lazy and do not return an xexpression, but rather an evaluated :cpp:type:`xt::xarray`
or :cpp:type:`xt::xtensor`.

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xmath.hpp>

    xt::xarray<double> a = xt::ones<double>({5, 8, 3});
    xt::xarray<double> res = xt::cumsum(a, 1);
    // => res.shape() = {5, 8, 3};
    // => res(0, 0, 0) = 1
    // => res(0, 7, 0) = 8

You can also call the :cpp:func:`xt::accumulate` generator with your own accumulating
function. For example, the implementation of cumsum is as follows:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xaccumulator.hpp>

    xt::xarray<double> arr = some_init_function({5, 5, 5});
    xt::xarray<double> res = xt::accumulate([](double a, double b) { return a + b; },
                                            arr,
                                            1);

Like reducers, accumulators accept a template parameter to specify the ``value_type``
of the initial value of the accumulation. The ``value_type`` of the result is computed
with the same rules as those for reducers:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xaccumulator.hpp>

    xt::xarray<int> arr = some_init_function({5, 5, 5});
    auto r1 = xt::cumsum<short>(a, 1);
    // r1 holds int values
    auto r2 = xt::cumsum<long int>(a, 1);
    // r2 hols long int values

Evaluation strategy
-------------------

Generally, *xtensor* implements a :ref:`lazy execution model <lazy-evaluation>`,
but under certain circumstances, a *greedy* execution model with immediate
execution can be favorable. For example, reusing (and recomputing) the same
values of a reducer over and over again if you use them in a loop can cost a
lot of CPU cycles. Additionally, *greedy* execution can benefit from SIMD
acceleration over reduction axes and is faster when the entire result needs to
be computed.

Therefore, xtensor allows to select an :cpp:enum:`xt::evaluation_strategy`. Currently, two
evaluation strategies are implemented: :cpp:enumerator:`xt::evaluation_strategy::immediate` and
:cpp:enumerator:`xt::evaluation_strategy::lazy`.
When :cpp:enumerator:`~xt::evaluation_strategy::immediate` evaluation is selected, the
return value is not an xexpression, but an in-memory datastructure such as a
xarray or xtensor (depending on the input values).

Choosing an evaluation_strategy is straightforward. For reducers:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xreducer.hpp>

    xt::xarray<double> a = xt::ones<double>({3, 2, 4, 6, 5});
    auto res = xt::sum(a, {1, 3}, xt::evaluation_strategy::immediate);
    // or select the default:
    // auto res = xt::sum(a, {1, 3}, xt::evaluation_strategy::lazy);

Note: for accumulators, only the :cpp:enumerator:`~xt::evaluation_strategy::immediate` evaluation
strategy is currently implemented.

Universal functions and vectorization
-------------------------------------

*xtensor* provides utilities to **vectorize any scalar function** (taking
multiple scalar arguments) into a function that will perform on
:cpp:type:`xt::xexpression` s, applying the lazy broadcasting rules which we described in a
previous section. These functions are called :cpp:type:`xt::xfunction` s.
They are *xtensor*'s counterpart to numpy's universal functions.

Actually, all arithmetic and logical operators, inequality operator and
mathematical functions we described before are :cpp:type:`xt::xfunction` s.

The following snippet shows how to vectorize a scalar function taking two
arguments:

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xvectorize.hpp>

    int f(int a, int b)
    {
        return a + 2 * b;
    }

    auto vecf = xt::vectorize(f);
    xt::xarray<int> a = { 11, 12, 13 };
    xt::xarray<int> b = {  1,  2,  3 };
    xt::xarray<int> res = vecf(a, b);
    // => res = { 13, 16, 19 }
