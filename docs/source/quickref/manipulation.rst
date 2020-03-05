.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Manipulation
============

atleast_Nd
----------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a0 = 123;
    auto r1 = xt::atleast_1d(a0);

    xt::xarray<int> a1 = { 1, 2, 3 };
    auto r2 = xt::atleast_2d(a1);
    auto r3 = xt::atleast_3d(a1);
    auto r4 = xt::atleast_Nd<4>(a1);

expand_dims
-----------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto r0 = xt::expand_dims(a, 0);
    auto r1 = xt::expand_dims(a, 1);
    auto r2 = xt::expand_dims(a, 2);

flip
----

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto f0 = xt::flip(a, 0);
    auto f1 = xt::flip(a, 1);

repeat
------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2}, {3, 4}};
    auto r0 = xt::repeat(a, 3, 1);
    auto r1 = xt::repeat(a, {1, 2}, 0);

roll
----

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto t0 = xt::roll(a, 2);
    auto t1 = xt::roll(a, 2, 1);

rot90
-----

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto r0 = xt::rot90<1>(a);
    auto r1 = xt::rot90<-2>(a);
    auto r2 = xt::rot90(a);
    auto r4 = xt::rot90(a, {-2, -1});
    
split
-----

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto s0 = xt::split(a, 3);
    auto s1 = xt::split(a, 3, 1);

hsplit
------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    auto res = xt::hsplit(a, 2);
    
vsplit
------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    auto res = xt::vsplit(a, 2);

squeeze
-------

.. code::

    #include "xmanipulation.hpp"

    auto b = xt::xarray<double>::from_shape({3, 3, 1, 1, 2, 1, 3});
    auto sq0 = xt::xqueeze(b);
    auto sq1 = squeeze(b, {2, 3}, check_policy::full());
    auto sq2 = squeeze(b, 2);

trim_zeros
----------

.. code::

    #include "xmanipulation.hpp"

    xt::xarray<int> a = {0, 0, 0, 1, 3, 0};
    auto t0 = xt::trim_zeros(a);
    auto t1 = xt::trim_zeros(a, "b");
    auto t2 = xt::trim_zeros(a, "f");

