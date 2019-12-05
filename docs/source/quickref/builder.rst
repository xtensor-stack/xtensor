.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Builders
========

Most of ``xtensor`` builders return unevaluated expressions (see :ref:`lazy-evaluation`
for more details) that can be assigned to any kind of ``xtensor`` container.

Ones
----

.. code::

    // Lazy version
    auto e = xt::ones<double>({2, 3});
    std::cout << e < std::endl;
    // Outputs {{1., 1., 1.}, {1., 1., 1.}}

    // Evaluated versions
    using fixed_tensor = xt::xtensor_fixed<double, xt::xshape<2, 3>>;
    xt::xarray<double>     a0 = xt::ones<double>({2, 3});
    xt::xtensor<double, 2> a1 = xt::ones<double>({2, 3});
    fixed_tensor           a2 = xt::ones<double>({2, 3});

Zeros
-----

.. code::

    // Lazy version
    auto e = xt::zeros<double>({2, 3});
    std::cout << e << std::endl;
    // Outputs {{0., 0., 0.}, {0., 0., 0.}}

    // Evaluated versions
    using fixed_tensor = xt::xtensor_fixed<double, xt::xshape<2, 3>>;
    xt::xarray<double>     a0 = xt::zeros<double>({2, 3});
    xt::xtensor<double, 2> a1 = xt::zeros<double>({2, 3});
    fixed_tensor           a2 = xt::zeros<double>({2, 3});

Empty
-----

``xt::empty`` creates a container of uninitialized values. It selects the best container
match from the supplied shape:

.. code::

    xt::xarray<double>::shape_type sh0 = {2, 3};
    auto a0 = xt::empty<double>(sh0);
    // a0 is xt::xarray<double>

    xt::xtensor<double, 2>::shape_type sh1 = {2, 3};
    auto a1 = xt::empty<double>(sh1);
    // a1 is xt::xtensor<double, 2>

    xt::xshape<2, 3> sh2;
    auto a2 = xt::empty<double>(sh2);
    // a2 is xt::xtensor_fixed<double, xt::xshape<2, 3>>

Full like
---------

``xt::full_like`` returns a container with the same shape as the input expression, and
filled with the specified value:

.. code::

    xt::xarray<double> a0 = {{1., 2., 3.}, {4., 5., 6.}};
    auto b0 = xt::full_like(a0, 3.);
    std::cout << b0 << std::endl;
    // Outputs {{3., 3., 3.}, {3., 3., 3.}}
    // b0 is an xt::xarray<double>

    xt::xtensor<double, 2> a1 = {{1., 2., 3.}, {4., 5., 6.}};
    auto b1 = xt::full_like(a1, 3.);
    std::cout << b1 << std::endl;
    // Outputs {{3., 3., 3.}, {3., 3., 3.}}
    // b1 is an xt::xtensor<double, 2>

    xt::xtensor_fixed<double, xt::xshape<2, 3>> a2 = {{1., 2., 3.}, {4., 5., 6.}};
    auto b2 = xt::full_like(a2, 3.);
    std::cout << b2 << std::endl;
    // Outputs {{3., 3., 3.}, {3., 3., 3.}}
    // b2 is an xt::xtensor_fixed<double, xt::xshape<2, 3>>

Ones like
---------

``ones_like(e)`` is equivalent to ``full_like(e, 1.)``.

Zeros like
----------

``zeros_like(e)`` is equivalent to ``full_like(e, 0.)``.

Eye
---

Generates an array with ones on the specified diagonal:

.. code::

    auto a = xt::eye<double>({2, 3}, 1);
    std::cout << a << std::endl;
    // Outputs {{O, 1, 0}, {0, 0, 1}}

    auto b = xt::eye<double>({3, 2}, -1);
    std::cout << b << std::endl;
    // Outputs {{0, 0}, {1, 0}, {0, 1}}

    aut c = xt::eye<double>(3, 1);
    std::cout << c << std::endl;
    // Outputs {{O, 1, 0}, {0, 0, 1}, {0, 0, 0}}

Arange
------

Generates evenly spaced numbers:

.. code::

    auto e = xt::arange<double>(0., 10., 2);
    std::cout << e << std::endl;
    // Outputs {0., 2., 4., 6., 8.}

A common pattern is to use ``arange`` followed by reshape to initialize
a tensor with an arbitrary number of dimensions:

.. code::

    xt::xarray<double> a = xt::arange<double>(0., 6.).reshape({2, 3});
    std::cout << a << std::endl;
    // Outputs {{0., 1., 2.}, {3., 4., 5.}}

Linspace
--------

.. code::

    auto a = xt::linspace<double>(0., 10., 5);
    std::cout << a << std::endl;
    // Outputs {0., 2.5, 5., 7.5, 10.}

Logspace
--------

Similar to ``linspace`` but numbers are evenly space on a log scale.

Concatenate
-----------

.. code::

    xt::xarray<double> a = {{1, 2, 3}};
    xt::xarray<double> b = {{2, 3, 4}};

    auto c0 = xt::concatenate(xt::xtuple(a, b));
    std::cout << c0 << std::endl;
    // Outputs {{1, 2, 3}, {2, 3, 4}}

    auto c1 = xt::concatenate(xt::xtuple(a, b), 1);
    std::cout << c1 << std::endl;
    // Outputs {1, 2, 3, 2, 3, 4}

Stack
-----

``stack`` always creates a new dimension along which elements are stacked:

.. code::

    xt::xarray<double> a = {1, 2, 3};
    xt::xarray<double> b = {5, 6, 7};

    auto s0 = xt::stack(xt::xtuple(a, b));
    std::cout << s0 << std::endl;
    // Outputs {{1, 2, 3}, {5, 6, 7}}

    auto s1 = xt::stack(xt::xtuple(a, b), 1);
    std::cout << s1 << std::endl;
    // Outputs {{1, 5}, {2, 6}, {3, 7}}

HStack
------

.. code::

    xt::xarray<double> a0 = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<double> b0 = {{7, 8}, {9, 10}};
    auto c0 = xt::hstack(xt::xtuple(a0, b0));
    std::cout << c0 << std:endl;
    // Outputs {{1, 2, 3, 7, 8}, {4, 5, 6, 0, 10}}

    xt::xarray<double> a1 = {1, 2, 3};
    xt::xarray<double> b1 = {2, 3 ,4};
    auto c1 = xt::hastack(xt::xtuple(a1, b1));
    std::cout << c1 << std::endl;
    // Outputs {1, 2, 3, 2, 3, 4}

VStack
------

.. code::

    xt::xarray<double> a0 = {1, 2, 3};
    xt::xarray<double> b0 = {2, 3, 4};
    auto c0 = xt::vstack(xt::xtuple(a0, b0));
    std::cout << c0 << std::endl;
    // Outputs {{1, 2, 3}, {2, 3 ,4}}

    xt::xarray<double> a1 = {{1, 2, 3}, {4, 5 ,6}, {7, 8, 9}};
    xt::xarray<double> b1 = {{10, 11, 12}};
    auto c1 = xt::vstack(xt::xtuple(a1, b1));
    std::cout << c1 << std::endl;
    // Outputs {{1, 2, 3}, {4, 5 ,6}, {7, 8, 9}, {10, 11, 12}}

Diag
----

Returns a 2D-expression using the input value as its diagonal:

.. code::

    xt::xarray<double> a = {1, 5, 7};
    auto b = xt::diag(a);
    std::cout << b << std::endl;
    // Outputs {{1, 0, 0} {0, 5, 0}, {5, 0, 7}}

Diagonal
--------

Returns the elements on the diagonal of the expression

.. code::

    xt::xarray<double> a = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};
    auto d = xt::diagonal(a);
    std::cout << d << std::endl;
    // Outputs {1, 5, 9}



