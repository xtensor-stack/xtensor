.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _view-description:

Views
=====

Views are used to adapt the shape of an ``xexpression`` without changing it, nor copying it. Views are 
convenient tools for assigning parts of an expression: since they do not copy the underlying expression,
assigning to the view actually assigns to the underlying expression. `xtensor` provides many kinds of views.

Sliced views
------------

Sliced views consist of the combination of the ``xexpression`` to adapt, and a list of ``slice`` that specify how
the shape must be adapted. Sliced views are implemented by the ``xview`` class. Objects of this type should not be
instantiated directly, but though the ``view`` helper function.

Slices can be specified in the following ways:

- selection in a dimension by specifying an index (unsigned integer)
- ``range(min, max)``, a slice representing the interval [min, max)
- ``range(min, max, step)``, a slice representing the stepped interval [min, max)
- ``all()``, a slice representing all the elements of a dimension
- ``newaxis()``, a slice representing an additional dimension of length one
- ``keep(i0, i1, i2, ...)`` a slice selecting non-contiguous indices to keep on the underlying expression
- ``drop(i0, i1, i2, ...)`` a slice selecting non-contiguous indices to drop on the underlying expression

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

    // View with non contiguous slices
    auto v4 = xt::view(a, xt::drop(0), xt::all(), xt::keep(0, 3));
    // => v4.shape() = { 2, 2, 2 }
    // => v4(0, 0, 0) = a(1, 0, 0)
    // => v4(1, 1, 1) = a(2, 1, 3)

The range function supports the placeholder ``_`` syntax:

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xview.hpp"

    using namespace xt::placeholders;  // required for `_` to work

    auto a = xt::xarray<int>::from_shape({3, 2, 4});
    auto v1 = xt::view(a, xt::range(_, 2), xt::all(), xt::range(1, _));
    // The previous line is equivalent to
    auto v2 = xt::view(a, xt::range(0, 2), xt::all(), xt::range(1, 4));

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


Strided views
-------------

While the ``xt::view`` is a compile-time static expression, xtensor also contains a dynamic strided view in ``xstrided_view.hpp``.
The strided view and the slice vector allow to dynamically push_back slices, so when the dimension is unknown at compile time, the slice
vector can be built dynamically at runtime. Note that the slice vector is actually a type-alias for a ``std::vector`` of a ``variant`` for
all the slice types. The strided view does not support the slices returned by the ``keep`` and ``drop`` functions.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstrided_view.hpp"

    auto a = xt::xarray<int>::from_shape({3, 2, 3, 4, 5});

    xt::xstrided_slice_vector sv({xt::range(0, 1), xt::newaxis()});
    sv.push_back(1);
    sv.push_back(xt::all());

    auto v1 = xt::strided_view(a, sv);
    // v1 has the same behavior as the static view

    // Equivalent but shorter
    auto v2 = xt::strided_view(a, { xt::range(0, 1), xt::newaxis(), 1, xt::all() });
    // v2 == v1

    // ILLEGAL:
    auto v2 = xt::strided_view(a, { xt::all(), xt::all(), xt::all(), xt::keep(0, 3), xt::drop(1, 4) });
    // xt::drop and xt::keep are not supported with strided views

Since ``xtensor 0.16.3``, a new range syntax can be used with strided views:

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstrided_view.hpp"

    using namespace xt::placeholders;

    auto a = xt::xarray<int>::from_shape({3, 2, 3, 4, 5});
    auto v1 = xt::strided_view(a, {_r|0|1, 1, _r|_|2, _r|_|_|-1});
    // The previous line is equivalent to
    auto v2 = xt::strided_view(a, {xt::range(0, 1), 1, xt::range(_, 2), xt::range(_, _, -1)});

The ``xstrided_view`` is very efficient on contigous memory (e.g. ``xtensor`` or ``xarray``) but less efficient on xexpressions.

Transposed views
----------------

``xtensor`` provides a lazy transposed view on any expression, whose layout is either row major order or column major order. Trying to build
a transposed view on a expression with a dynamic layout throws an exception.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstrided_view.hpp"

    xt::xarray<int> a = { {0, 1, 2}, {3, 4, 5} };
    auto tr = xt::transpose(a);
    // tr == { {0, 3}, {1, 4}, {2, 5} }

    xt::xarray<int, layout_type::dynamic> b = { {0, 1, 2}, {3, 4, 5} };
    auto tr2 = xt::transpose(b);
    // => throw transpose_error

Like the strided view, the transposed view is built upon the ``xstrided_view``.

Flatten views
-------------

It is sometimes useful to have a one-dimensional view of all the elements of an expression. ``xtensor`` provides two functions
for that, ``ravel`` and ``flatten``. The former one let you specify the order used to read the elements while the latter one
uses the layout of the expression.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstrided_view.hpp"

    xt::xarray<int> a = { {0, 1, 2}, {3, 4, 5} };
    auto flc = xt::ravel<layout_type::column_major>(a);
    std::cout << flc << std::endl;
    // => prints { 0, 3, 1, 4, 2, 5 }

    auto fl = xt::flatten(a);
    std::cout << fl << std::endl;
    // => prints { 0, 1, 2, 3, 4, 5 }

Like the strided view and the transposed view, the flatten view is built upon the ``xstrided_view``.

Reshape views
-------------

The reshape view allows to handle an expression as if it was given a new shape, however no additional memory allocation occurs,
the original expression keeps its shape. Like any view, the underlying expression is not copied, thus assigning a value through
the view modifies the underlying expression.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xstrided_view.hpp"

    auto a = xt::xarray<int>::from_shape({3, 2, 4});
    auto v = xt::reshape_view(a, { 4, 2, 3 });
    // a(0, 0, 3) == v(0, 1, 0)
    // a(0, 1, 0) == v(0, 1, 1)

    v(0, 2, 0) = 4;
    // a(0, 1, 2) == 4

Like the strided view and the transposed view, the reshape view is built upon the ``xstrided_view``.

Dynamic views
-------------

The dynamic view is like the strided view, but with support of the slices returned by the ``keep`` and ``drop`` functions.
However, this support has a cost and the dynamic view is slower than the strided view, even when no keeping or dropping
slice is involved.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xdynamic_view.hpp

    auto a = xt::xarray<int>::from_shape({3, 2, 3, 4, 5});
    xt::xdynamic_slice_vector sv({xt::range(0, 1), xt::newaxis()});
    sv.push_back(1);
    sv.push_back(xt::all());
    sv.push_back(xt::keep(0, 2, 3));
    sv.push_back(xt::drop(1, 2, 4));

    auto v1 = xt::dynamic_view(a, sv});

    // Equivalent but shorter
    auto v2 = xt::dynamic_view(a, { xt::range(0, 1), xt::newaxis(), 1, xt::all(), xt::keep(0, 2, 3), xt::drop(1, 2, 4) });
    // v2 == v1

Index views
-----------

Index views are one-dimensional views of an ``xexpression``, containing the elements whose positions are specified by a list
of indices. Like for sliced views, the elements of the underlying ``xexpression`` are not copied. Index views should be built
with the ``index_view`` helper function.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xindex_view.hpp"

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
    #include "xtensor/xindex_view.hpp"

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
    #include "xtensor/xindex_view.hpp"

    xt::xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
    filtration(a, a >= 5) += 100;
    // => a = {{1, 105, 3}, {4, 105, 106}}

Masked view
-----------

Masked views are multidimensional views that apply a mask on an ``xexpression``.

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xmasked_view.hpp"

    xt::xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
    xt::xarray<bool> mask = {{true, false, false}, {false, true, false}};

    auto m = xt::masked_view(a, mask);
    // => m = {{1, masked, masked}, {masked, 5, masked}}

    m += 100;
    // => a = {{101, 5, 3}, {4, 105, 6}}

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

Assigning to a view
-------------------

When assigning an expression ``rhs`` to a container such as ``xarray``, this last one is resized so its shape is the same as the one
of ``RHS``. However, since views *cannot be resized*, when assigning an expression to a view, broadcasting rules are applied:

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xview.hpp"

    xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
    double b = 1.2;
    auto tr = view(a, 0, all());
    tr = b;
    // => a = {{1.2, 1.2, 1.2}, {3., 4., 5.}}
