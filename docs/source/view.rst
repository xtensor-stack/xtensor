.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Views
=====

Views are used to adapt the shape of an ``xexpression`` without changing it, nor copying it. `xtensor`
provides two kinds of views.

Sliced views
------------

Sliced views consist of the combination of the ``xexpression`` to adapt, and a list of ``slice`` s that specify how
the shape must be adapted. Sliced views are implemented by the ``xview`` class. Objects of this type should not be
instantiated directly, but though the ``view`` helper function.

Slices can be specified in the following ways:

- selection in a dimension by specifying an index (unsigned integer)
- ``range(min, max)``, a slice representing an interval
- ``range(min, max, step)``, a slice representing a stepped interval
- ``all()``, a slice representing all the elements of a dimension
- ``newaxis()``, a slice representing an additional dimension of length one

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xview.hpp"

    std::vector<size_t> shape = {3, 2, 4};
    xt::xarray<int> a(shape);
    
    // View with same number of dimensions
    auto v1 = xt::view(a, xt::range(1, 3), xt:all(), xt::range(1, 3));
    // => v1.shape() = { 2, 2, 2 }
    // => v1(0, 0, 0) = a(1, 0, 1)
    // => v1(1, 1, 1) = a(2, 1, 2)

    // View reducing the number of dimensions
    auto v2 = xt::view(a, 1, xt::all(), xt::range(0, 4, 2));
    // => v1.shape() = { 2, 2 }
    // => v1(0, 0) = a(1, 0, 0)
    // => v1(1, 1) = a(1, 1, 2)

    // View increasing the number of dimensions
    auto v3 = xt::view(a, xt::all(), xt::all(), xt::newaxis(), xt::all());
    // => v1.shape() = { 3, 2, 1, 4 }
    // => v1(0, 0, 0, 0) = a(0, 0, 0)

``xview`` does not perform a copy of the underlying expression. This means if you modify an element of the ``xview``,
you are actually also altering the underlying expression.

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xview.hpp"

    std::vector<size_t> shape = {3, 2, 4};
    xt::xarray<int> a(shape, 0);

    auto v1 = xt::view(a, 1, xt::all(), xt::range(1, 3));
    v1(0, 0) = 1;
    // => a(1, 0, 1) = 1

Broadcasting views
------------------

The second type of view provided by `xtensor` is *broadcasting view*. Such a view broadcast an expression to the specified
shape. As long as the view is not assigned to an array, no memory allocation or copy occurs. Broadcasting views should be
built with the ``broadcast`` helper function.

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xbroadcast.hpp"

    std::vector<size_t> s1 = { 2, 3 };
    std::vector<size_t> s2 = { 3, 2, 3 };

    xt::array<int> a1(s1);
    auto bv = xt::broadcast(a1, s2);
    // => bv(0, 0, 0) = bv(1, 0, 0) = bv(2, 0, 0) = a(0, 0)


