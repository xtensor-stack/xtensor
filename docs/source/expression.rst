.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _lazy-evaluation:


Expressions and lazy evaluation
===============================

*xtensor* is more than an N-dimensional array library: it is an expression engine that allows numerical computation on any object implementing the expression interface.
These objects can be in-memory containers such as :cpp:type:`xt::xarray\<T\>` and :cpp:type:`xt::xtensor\<T\>`, but can also be backed by a database or a representation on the file system.
This also enables creating adaptors as expressions for other data structures.

Expressions
-----------

Assume ``x``, ``y`` and ``z`` are arrays of *compatible shapes* (we'll come back to that later), the return type of an expression such as ``x + y * sin(z)`` is **not an array**.
The result is an :cpp:type:`xt::xexpression` which offers the same interface as an N-dimensional array but does not hold any value.
Such expressions can be plugged into others to build more complex expressions:

.. code::

    auto f = x + y * sin(z);
    auto f2 = w + 2 * cos(f);

The expression engine avoids the evaluation of intermediate results and their storage in temporary arrays, so you can achieve the same performance as if you had written
a simple loop. Assuming ``x``, ``y`` and ``z`` are one-dimensional arrays of length ``n``,

.. code::

    xt::xarray<double> res = x + y * sin(z)

will produce quite the same assembly as the following loop:

.. code::

    xt::xarray<double> res(n);
    for(size_t i = 0; i < n; ++i)
    {
        res(i) = x(i) + y(i) * sin(z(i));
    }

Lazy evaluation
---------------

An expression such as ``x + y * sin(z)`` does not hold the result. **Values are only computed upon access or when the expression is assigned to a container**. This
allows to operate symbolically on very large arrays and only compute the result for the indices of interest:

.. code::

    // Assume x and y are xarrays each containing 1 000 000 objects
    auto f = cos(x) + sin(y);

    double first_res = f(1200);
    double second_res = f(2500);
    // Only two values have been computed

That means if you use the same expression in two assign statements, the computation of the expression will be done twice. Depending on the complexity of the computation
and the size of the data, it might be convenient to store the result of the expression in a temporary variable:

.. code::

    // Assume x and y are small arrays
    xt::xarray<double> tmp = cos(x) + sin(y);
    xt::xarray<double> res1 = tmp + 2 * x;
    xt::xarray<double> res2 = tmp - 2 * x;

Forcing evaluation
------------------

If you have to force the evaluation of an xexpression for some reason (for example, you want to have all results in memory to perform a sort or use external BLAS functions) then you can use :cpp:func:`xt::eval` on an xexpression.
Evaluating will either return a *rvalue* to a newly allocated container in the case of an xexpression, or a reference to a container in case you are evaluating a :cpp:type:`xt::xarray` or :cpp:type:`xt::xtensor`.
Note that, in order to avoid copies, you should use a universal reference on the lefthand side (``auto&&``).
For example:

.. code::

    xt::xarray<double> a = {1, 2, 3};
    xt::xarray<double> b = {3, 2, 1};
    auto calc = a + b; // unevaluated xexpression!
    auto&& e = xt::eval(calc); // a rvalue container xarray!
    // this just returns a reference to the existing container
    auto&& a_ref = xt::eval(a);

Broadcasting
------------

The number of dimensions of an :cpp:type:`xt::xexpression` and the sizes of these dimensions are provided by the :cpp:func:`~xt::xexpression::shape` method, which returns a sequence of unsigned integers
specifying the size of each dimension. We can operate on expressions of different shapes of dimensions in an elementwise fashion.
Broadcasting rules of *xtensor* are similar to those of NumPy_ and libdynd_.

In an operation involving two arrays of different dimensions, the array with the lesser dimensions is broadcast across the leading dimensions of the other.
For example, if ``A`` has shape ``(2, 3)``, and ``B`` has shape ``(4, 2, 3)``, the result of a broadcast operation with ``A`` and ``B`` has shape ``(4, 2, 3)``.

.. code:: none

       (2, 3) # A
    (4, 2, 3) # B
    ---------
    (4, 2, 3) # Result

The same rule holds for scalars, which are handled as 0-D expressions. If ``A`` is a scalar, the equation becomes:

.. code:: none

           () # A
    (4, 2, 3) # B
    ---------
    (4, 2, 3) # Result

If matched up dimensions of two input arrays are different, and one of them has size ``1``, it is broadcast to match the size of the other. Let's say B has the shape ``(4, 2, 1)``
in the previous example, so the broadcasting happens as follows:

.. code:: none

       (2, 3) # A
    (4, 2, 1) # B
    ---------
    (4, 2, 3) # Result

Accessing elements
------------------

You can access the elements of any :cpp:type:`xt::xexpression` with :cpp:func:`~xt::xexpression::operator()()`:

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};
    auto f = 2 * a;

    double d1 = a(0, 2);
    double d2 = f(1, 2);

It is possible to call :cpp:func:`~xt::xexpression::operator()()` with fewer or more arguments than the number of dimensions
of the expression:

- if :cpp:func:`~xt::xexpression::operator()()` is called with too many arguments, we drop the most left ones
- if :cpp:func:`~xt::xexpression::operator()()` is called with too few arguments, we prepend them with ``0`` values until
  we match the number of dimensions

.. code::

    #include <xtensor/xarray.hpp>

    xt::xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}};

    double d1 = a(2); // equivalent to a(0, 2)
    double d2 = a(1, 1, 2) // equivalent to a(1, 2)

The reason for this is that it is the one rule that ensures ``(a + b)(i0, ..., in) = a(i0, ..., in) + b(i0, ..., in)``,
i.e. commutativity of element access and broadcasting.

Expression interface
--------------------

All :cpp:type:`xt::xexpression` s in :cpp:type:`xt::xtensor` provide at least the following interface:

Shape
~~~~~

- :cpp:func:`~xt::xexpression::dimension`: returns the number of dimensions of the expression.
- :cpp:func:`~xt::xexpression::shape`: returns the shape of the expression.

.. code::

    #include <vector>
    #include <xtensor/xarray.hpp>

    using array_type = xt::xarray<double>;
    using shape_type = array_type::shape_type;
    shape_type shape = {3, 2, 4};
    array_type a(shape);
    size_t d = a.dimension();
    const shape_type& s = a.shape();
    bool res = (d == shape.size()) && (s == shape);
    // => res = true

Element access
~~~~~~~~~~~~~~

- :cpp:func:`~xt::xexpression::operator()()` is an access operator that can take multiple integral arguments or none.
- :cpp:func:`~xt::xexpression::at` is similar to :cpp:func:`~xt::xexpression::operator()()` but checks that its number
  of arguments does not exceed the number of dimensions, and performs bounds checking.
  This should not be used where you expect :cpp:func:`~xt::xexpression::operator()()` to perform broadcasting.
- :cpp:func:`~xt::xexpression::operator[]` has two overloads: one that takes a single integral argument and is
  equivalent to the call of :cpp:func:`~xt::xexpression::operator()()` with one argument, and one with a single
  multi-index argument, which can be of a size determined at runtime.
  This operator also supports braced initializer arguments.
- :cpp:func:`~xt::xexpression::element` is an access operator which takes a pair of iterators on a container of indices.
- :cpp:func:`~xt::xexpression::periodic` is the equivalent of :cpp:func:`~xt::xexpression::operator()()` that can deal
  with periodic indices (for example ``-1`` for the last item along an axis).
- :cpp:func:`~xt::xexpression::in_bounds` returns a ``bool`` that is ``true`` only if indices are valid for the array.

.. code::

    #include <vector>
    #inclde "xtensor/xarray.hpp"

    // xt::xarray<double> a = ...
    std::vector<size_t> index = {1, 1, 1};
    double v1 = a(1, 1, 1);
    double v2 = a[index],
    double v3 = a.element(index.begin(), index.end());
    // => v1 = v2 = v3

Iterators
~~~~~~~~~

- :cpp:func:`~xt::xexpression::begin` and :cpp:func:`~xt::xexpression::end` return instances of :cpp:type:`xt::xiterator`
  which can be used to iterate over all the elements of the expression.
  The layout of the iteration can be specified through the :cpp:enum:`xt::layout_type` template parameter, accepted values
  are :cpp:enumerator:`xt::layout_type::row_major` and :cpp:enumerator:`xt::layout_type::column_major`.
  If not specified, :c:macro:`XTENSOR_DEFAULT_TRAVERSAL` is used.
  This iterator pair permits to use algorithms of the STL with :cpp:type:`xt::xexpression` as if they were simple containers.
- :cpp:func:`begin(shape) <xt::xiterator xt::xexpression::begin(xt::xshape)>` and
  :cpp:func:`end(shape) <xt::xiterator xt::xexpression::end(xt::xshape)>` are similar but take a *broadcasting shape*
  as an argument.
  Elements are iterated upon in :c:macro:`XTENSOR_DEFAULT_TRAVERSAL` if no :cpp:enum:`xt::layout_type` template parameter
  is specified.
  Certain dimensions are repeated to match the provided shape as per the rules described above.
- :cpp:func:`~xt::xexpression::rbegin` and :cpp:func:`~xt::xexpression::rend` return instances of :cpp:type:`xt::xiterator`
  which can be used to iterate over all the elements of the reversed expression.
  As :cpp:func:`~xt::xexpression::begin` and :cpp:func:`~xt::xexpression::end`, the layout of the iteration can be
  specified through the :cpp:enum:`xt::layout_type` parameter.
- :cpp:func:`rbegin(shape) <xt::xiterator xt::xexpression::rbegin(xt::xshape)>` and
  :cpp:func:`rend(shape) <xt::xiterator xt::xexpression::rend(xt::xshape)>` are the reversed counterpart of
  :cpp:func:`begin(shape) <xt::xiterator xt::xexpression::begin(xt::xshape)>` and
  :cpp:func:`end(shape) <xiterator xt::xexpression::end(xt::xshape)>`.

.. _NumPy: http://www.numpy.org
.. _libdynd: http://libdynd.org
