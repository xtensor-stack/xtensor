.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Reductions
==========

Sum
---

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::sum(a, {1});
    std::cout << r0 << std::endl;
    // Outputs {6, 15}

    xt::xarray<int> r1 = xt::sum(a);
    std::cout << r1 << std::endl;
    // Outputs {21}, i.e. r1 is a 0D-tensor

    int r2 = xt::sum(a)();
    std::cout << r2 << std::endl;
    // Outputs 21
    
    auto r3 = xt::sum(a, {1});
    std::cout << r3 << std::endl;
    // Outputs {6, 15}, but r3 is an unevaluated expression
    // the values are computed upon each access

Prod
----

.. code::

    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> r0 = xt::prod(a, {1});
    xt::xarray<int> r1 = xt::prod(a);
    int r2 = xt::prod(a)();
    auto r3 = xt::prod(a, {0});

Mean
----

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::mean(a, {1});
    xt::xarray<int> r1 = xt::mean(a);
    int r2 = xt::mean(a)();
    auto r3 = xt::mean(a, {0});

Variance
--------

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::variance(a, {1});
    xt::xarray<int> r1 = xt::variance(a);
    int r2 = xt::variance(a)();
    auto r3 = xt::variance(a, {0});

Standard deviation
------------------

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::stddev(a, {1});
    xt::xarray<int> r1 = xt::stddev(a);
    int r2 = xt::stddev(a)();
    auto r3 = xt::stddev(a, {0});

Diff
----

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::diff(a, 1, {0});
    std::cout << r0 << std::endl;
    // Outputs {{1, 1}, {1, 1}}

Amax
----

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::amax(a, {1});
    std::cout << r0 << std::endl;
    // Outputs {3, 6}

Amin
----

.. code::

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> r0 = xt::amin(a, {0});
    std::cout << r0 << std::endl;
    // Outputs {1, 2, 3}

Norms
-----

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> b0 = xt::norm_l0(a, {1});
    xt::xarray<double> b1 = xt::norm_l1(a, {1});
    xt::xarray<double> b2 = xt::norm_sq(a, {1});
    xt::xarray<double> b3 = xt::norm_l2(a, {1});
    xt::xarray<double> b4 = xt::norm_linf(a, {1});
    xt::xarray<double> b5 = xt::norm_lp_to_p(a, {1});
    xt::xarray<double> b6 = xt::norm_lp(a, {1});
    xt::xarray<double> b7 = xt::norm_induced_l1(a, {1});
    xt::xarray<double> b8 = xt::norm_induced_linf(a, {1});

Accumulating functions
----------------------

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> b0 = xt::cumsum(a, {1});
    std::cout << b0 << std::endl;
    // Outputs {{1., 3., 6.}, {4., 9., 15.}}

    xt::xarray<double> b1 = xt::cumprod(a, {1});
    std::cout << b1 << std::endl;
    // Outputs {{1., 2., 6.}, {4., 20., 120.}}
