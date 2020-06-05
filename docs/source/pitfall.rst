.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Common pitfalls
===============

xarray initialization
---------------------

.. code::

    xt::xarray<double> a({1, 3, 4, 2});

does not initialize a 4D-array, but a 1D-array containing the values ``1``, ``3``,
``4``, and ``2``. 
It is strictly equivalent to

.. code::

    xt::xarray<double> a = {1, 3, 4, 2};

To initialize a 4D-array with the given shape, use the ``from_shape`` static method:

.. code::

    auto a = xt::xarray<double>::from_shape({1, 3, 4, 2});

The confusion often comes from the way ``xtensor`` can be initialized:

.. code::

    xt::xtensor<double, 4> a = {1, 3, 4, 2};

In this case, a 4D-tensor with shape ``(1, 3, 4, 2)`` is initialized.

Intermediate result
-------------------

Consider the following function:

.. code::

    template <class C>
    auto func(const C& c)
    {
        return (1 - func_tmp(c)) / (1 + func_tmp(c));
    }

where ``func_tmp`` is another unary function accepting an xtensor expression. You may
be tempted to simplify it a bit:

.. code::

    template <class C>
    auto func(const C& c)
    {
        auto tmp = func_tmp(c);
        return (1 - tmp) / (1 + tmp);
    }

Unfortunately, you introduced a bug; indeed, expressions in ``xtensor`` are not evaluated
immediately, they capture their arguments by reference or copy depending on their nature,
for future evaluation. Since ``tmp`` is an lvalue, it is captured by reference in the last
statement; when the function returns, ``tmp`` is destroyed, leading to a dangling reference
in the returned expression.

Replacing ``auto tmp`` with ``xt::xarray<double> tmp`` does not change anything, ``tmp``
is still an lvalue and thus captured by reference.

Random numbers not consistent
-----------------------------

Using a random number function from xtensor actually returns a lazy 
generator. That means, accessing the same element of a random number
generator does not give the same random number if called twice.

.. code::

    auto gen = xt::random::rand<double>({10, 10});
    auto a0 = gen(0, 0);
    auto a1 = gen(0, 0);

    // a0 != a1 !!!

You need to explicitly assign or eval a random number generator, 
like so:

.. code::

    xt::xarray<double> xr = xt::random::rand<double>({10, 10});
    auto xr2 = eval(xt::random::rand<double>({10, 10}));

    // now xr(0, 0) == xr(0, 0) is true.

variance arguments
------------------

When ``variance`` is passed an expression and an integer parameter, this latter
is not the axis along which the variance must be computed, but the degree of freedom:

.. code::

    xt::xtensor<double, 2> a = {{1., 2., 3.}, {4., 5., 6.}};
    std::cout << xt::variance(a, 1) << std::endl;
    // Outputs 3.5

If you want to specify an axis, you need to pass an initializer list:

.. code::

    xt::xtensor<double, 2> a = {{1., 2., 3.}, {4., 5., 6.}};
    std::cout << xt::variance(a, {1}) << std::endl;
    .. Outputs { 0.666667, 0.666667 }

fixed_shape on Windows
----------------------

Builder functions such as ``empty`` or ``ones`` accept an initializer list
as argument. If the elements of this list do not have the same type, a
curious compilation error may occur on Windows:

.. code::

    size_t N = 10ull;
    xt::xarray<int> ages = xt::empty<int>({N, 4ul});

    // error: cannot convert argument 1 from 'initializer list'
    // to 'const xt::fixed_shape<> &'

To avoid this compiler bug (for which we don't have a workaround), ensure
all the elements in the initializer list have the same type.
