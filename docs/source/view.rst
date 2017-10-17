.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

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
    auto v1 = xt::view(a, xt::range(1, 3), xt::all(), xt::range(1, 3));
    // => v1.shape() = { 2, 2, 2 }
    // => v1(0, 0, 0) = a(1, 0, 1)
    // => v1(1, 1, 1) = a(2, 1, 2)

    // View reducing the number of dimensions
    auto v2 = xt::view(a, 1, xt::all(), xt::range(0, 4, 2));
    // => v2.shape() = { 2, 2 }
    // => v2(0, 0) = a(1, 0, 0)
    // => v2(1, 1) = a(1, 1, 2)

    // View increasing the number of dimensions
    auto v3 = xt::view(a, xt::all(), xt::all(), xt::newaxis(), xt::all());
    // => v3.shape() = { 3, 2, 1, 4 }
    // => v3(0, 0, 0, 0) = a(0, 0, 0)

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


Dynamic views
-------------

While the ``xt::view`` is a compile-time static expression, xtensor also contains a dynamic view in ``xstridedview.hpp``. The dynamic view and the slice vector allow to dynamically push_back slices, so when the dimension is unknown at compile time, the slice vector can be built dynamically at runtime. All the same slices as in xview can be used.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstridedview.hpp"

    auto a = xt::xarray<int>::from_shape({3, 2, 3, 4, 5});

    // note that `a` has to be passed into the slice_vector constructor
    xt::slice_vector sv(a, xt::range(0, 1), xt::newaxis());
    sv.push_back(1);
    sv.push_back(xt::all());
    // there is also a shorthand syntax: sv.append(1, xt::all());

    auto v1 = xt::dynamic_view(a, sv);
    // v1 has the same behavior as the static view

The dynamic_view is implemented on top of the strided_view, which is very efficient on contigous memory (e.g. xtensor or xarray) 
but less efficient on xexpressions.


Index views
-----------

Index views are one-dimensional views of an ``xexpression``, containing the elements whose positions are specified by a list
of indices. Like for sliced views, the elements of the underlying ``xexpression`` are not copied. Index views should be built
with the ``index_view`` helper function.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xindexview.hpp"

    xt::xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
    auto b = xt::index_view(a, {{0,0}, {1, 0}, {0, 1}});
    // => b = { 1, 4, 5 }
    b += 100;
    // => a = {{101, 5, 3}, {104, 105, 6}}

Filter views
------------

Filters are one-dimensional views holding elements of an ``xexpression`` that verify a given condition. Like for other views,
the elements of the underlying ``xexpression`` are not copied. Filters should be built with the ``filter`` helper function.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xindexview.hpp"

    xt::xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
    auto v = xt::filter(a, a >= 5);
    // => v = { 5, 5, 6 }
    v += 100;
    // => a = {{1, 105, 3}, {4, 105, 106}}

Filtration
----------

Sometimes, the only thing you want to do with a filter is to assign it a scalar. Though this can be done as shown
in the previous section, this is not the *optimal* way to do it. `xtensor` provides a specially optimized mechanism
for that, called filtration. A filtration IS NOT an ``xexpression``, the only methods it provides are scalar and 
computed scalar assignments.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xindexview.hpp"

    xt::xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
    filtration(a, a >= 5) += 100;
    // => a = {{1, 105, 3}, {4, 105, 106}}

Broadcasting views
------------------

Another type of view provided by `xtensor` is *broadcasting view*. Such a view broadcast an expression to the specified
shape. As long as the view is not assigned to an array, no memory allocation or copy occurs. Broadcasting views should be
built with the ``broadcast`` helper function.

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xbroadcast.hpp"

    std::vector<size_t> s1 = { 2, 3 };
    std::vector<size_t> s2 = { 3, 2, 3 };

    xt::xarray<int> a1(s1);
    auto bv = xt::broadcast(a1, s2);
    // => bv(0, 0, 0) = bv(1, 0, 0) = bv(2, 0, 0) = a(0, 0)

Complex views
-------------

In the case of tensor containing complex numbers, `xtensor` provides views returning ``xexpression`` corresponding to the real
and imaginary parts of the complex numbers. Like for other views, the elements of the underlying ``xexpression`` are not copied.

Functions ``xt::real`` and ``xt::imag`` respectively return views on the real and imaginary part of a complex expression.
The returned value is an expression holding a closure on the passed argument.

- The constness and value category (rvalue / lvalue) of ``real(a)`` is the same as that of ``a``. Hence, if ``a`` is a non-const lvalue,
  ``real(a)`` is an non-const lvalue reference, to which one can assign a real expression.
- If ``a`` has complex values, the same holds for ``imag(a)``. The constness and value category of ``imag(a)`` is the same as that of ``a``.
- If ``a`` has real values, ``imag(a)`` returns ``zeros(a.shape())``.

.. code::

    #include <complex>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xcomplex.hpp"

    using namespace std::complex_literals;

    xarray<std::complex<double>> e =
        {{1.0       , 1.0 + 1.0i},
         {1.0 - 1.0i, 1.0       }};

    real(e) = zeros<double>({2, 2});
    // => e = {{0.0, 0.0 + 1.0i}, {0.0 - 1.0i, 0.0}};
  
