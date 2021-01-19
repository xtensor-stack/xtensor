.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Basics
======

Tensor types
------------

- ``xarray<T>``: tensor that can be reshaped to any number of dimensions.
- ``xtensor<T, N>``: tensor with a number of dimensions set to ``N`` at compile time.
- ``xtensor_fixed<T, xshape<I, J, K>``: tensor whose shape is fixed at compile time.
- ``xchunked_array<CS>``: chunked array using the ``CS`` chunk storage.

.. note::

   Except if mentioned otherwise, the methods described below are available for the
   three kinds of containers, even if the examples show ``xarray`` usage only.

Initialization
--------------

Tensor with dynamic shape:

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<double>::shape_type shape = {2, 3};
    xt::xarray<double> a0(shape);
    xt::xarray<double> a1(shape, 2.5);
    xt::xarray<double> a2 = {{1., 2., 3.}, {4., 5., 6.}};
    auto a3 = xt::xarray<double>::from_shape(shape);

Tensor with static number of dimensions:

.. code::

    #include <xtensor/xtensor.hpp>

    xt::xtensor<double, 2>::shape_type shape = {2, 3};
    xt::xtensor<double, 2> a0(shape);
    xt::xtensor<double, 2> a1(shape, 2.5);
    xt::xtensor<double, 2> a2 = {{1., 2., 3.}, {4., 5., 6.}};
    auto a3 = xt::xtensor<double, 2>::from_shape(shape);

Tensor with fixed shape:

.. code::

    #include <xtensor/xfixed.hpp>

    xt::xtensor_fixed<double, xt::xshape<2, 3>> = {{1., 2., 3.}, {4., 5., 6.}};

In-memory chunked tensor with dynamic shape:

.. code::

    #include <xtensor/xchunked_array.hpp>

    std::vector<std::size_t> shape = {10, 10, 10};
    std::vector<std::size_t> chunk_shape = {2, 3, 4};
    auto a = xt::chunked_array<double>(shape, chunk_shape);

Output
------

.. code::

    #include <xtensor/xarray.hpp>
    #include <xtensor/xfixed.hpp>
    #include <xtensor/xio.hpp>
    #include <xtensor/xtensor.hpp>

    xt::xarray<double> a = {{1., 2.}, {3., 4.}};
    std::cout << a << std::endl;

    xt::xtensor<double, 2> b = {{1., 2.}, {3., 4.}};
    std::cout << b << std::endl;

    xt::xtensor_fixed<double, xt::xshape<2, 2>> c = {{1., 2.}, {3., 4.}};
    std::cout << c << std::endl;

Shape - dimension - size
------------------------

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    auto size = a.size();     // size = 6
    auto dim = a.dimension(); // dim = 2
    auto shape = a.shape();   // shape = {2, 3}
    auto sh1 = a.shape(1);    // sh1 = 3

Print the shape
---------------

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    auto shape = a.shape();
    std::cout << xt::adapt(shape) << std::endl;

Reshape
-------

The number of elements of an ``xarray`` must remain the same:

.. code::

    xt::xarray<double> a0 = {1., 2., 3., 4., 5., 6.};
    a0.reshape({2, 3});
    std::cout << a0 << std::endl;
    // outputs {{1., 2., 3.}, {4., 5., 6. }}

For ``xtensor`` the number of elements and the number of dimensions
must remain the same:

.. code::

    xt::xtensor<double, 2> a1 = {{1., 2.}, {3., 4.}, {5., 6.}};
    a1.reshape({2, 3});
    std::cout << a1 << std::endl;
    // outputs {{1., 2., 3.}, {4., 5., 6. }}

One value in the shape can be -1. In this case, the value is inferred from the
length of the underlying buffer and remaining dimensions:

.. code::

    xt::xarray<double> a0 = {1., 2., 3., 4., 5., 6.};
    a0.reshape({2, -1});
    std::cout << a0 << std::endl;
    // outputs {{1., 2., 3.}, {4., 5., 6. }}

    xt::xtensor<double, 2> a1 = {{1., 2.}, {3., 4.}, {5., 6.}};
    a1.reshape({-1, 3});
    std::cout << a1 << std::endl;
    // outputs {{1., 2., 3.}, {4., 5., 6. }}

``reshape`` is not defined for ``xtensor_fixed``.

Resize
------

.. code::

    xt::xarray<double> a0 = {1., 2., 3, 4.};
    a0.resize({2, 3});

When resizing an ``xtensor`` object, the number of dimensions must remain
the same:

.. code::

    xt::xtensor<double, 2> a1 = {{1., 2.}, {3., 4.}};
    a1.resize({2, 3});

``resize`` is not defined for ``xtensor_fixed``.

.. warning::

    Contrary to STL containers like std::vector, resize do NOT
    preserve elements.

Element access
--------------

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    double d0 = a(0, 2);   // d0 is 3.
    double d1 = a(2);      // d1 is a(0, 2)
    double d2 = a[{0, 2}]; // d2 is a(0, 2)

The same operators are used for writing values:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    a(0, 2)   = 8.;
    a(2)      = 8.;
    a[{0, 2}] = 8.;

The ``at`` method is an access operator with bound checking:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    double d0 = a.at(0, 3);   // throws
    double d1 = a.at(3);      // throws

The ``periodic`` method is an access operator that applies periodicity
to its arguments:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    double d0 = a.periodic(2, -1); // d0 is 3

Fill
----

.. code::

    auto a = xt::xarray<double>::from_shape({2, 3});
    a.fill(2.);
    std::cout << a << std::endl;
    // Outputs {{2., 2., 2.}, {2., 2., 2.}}

Iterators
---------

``xtensor`` containers provide iterators compatible with algorithms from the STL:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> b(a.shape());
    std::transform(a.cbegin(), a.cend(), b.begin(), [](auto&& v) { return v + 1; });
    std::cout << b << std::endl;
    // Outputs {{2., 3., 4.}, {5., 6., 7.}}

Reverse iterators are also available:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    xt::xarray<double> b(a.shape());
    std::copy(a.crbegin(), a.crend(), b.begin());
    std::cout << b << std::endl;
    // Outputs {{6., 5., 4.}, {3., 2., 1.}}
 
Data buffer
-----------

The underlying 1D data buffer can be accessed with the ``data`` method:

.. code::

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    a.data()[4] = 8.;
    std::cout << a << std::endl;
    // Outputs {{1., 2., 3.}, {8., 5., 6.}}
