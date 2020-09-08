.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Arrays and tensors
==================

Internal memory layout
----------------------

A multi-dimensional array of `xtensor` consists of a contiguous one-dimensional buffer combined with an indexing scheme that maps
unsigned integers to the location of an element in the buffer. The range in which the indices can vary is specified by the
`shape` of the array.

The scheme used to map indices into a location in the buffer is a strided indexing scheme. In such a scheme, the index ``(i0, ..., in)`` corresponds to the offset ``sum(ik * sk)`` from the beginning of the one-dimensional buffer, where ``(s0, ..., sn)`` are the `strides` of the array. Some particular cases of strided schemes implement well-known memory layouts:

- the row-major layout (or C layout) is a strided index scheme where the strides grow from right to left
- the column-major layout (or Fortran layout) is a strided index scheme where the strides grow from left to right

``xtensor`` provides a ``layout_type`` enum that helps to specify the layout used by multidimensional arrays. This enum can be used in two ways:

- at compile time, as a template argument. The value ``layout_type::dynamic`` allows specifying any strided index scheme at runtime (including row-major and column-major schemes), while ``layout_type::row_major`` and ``layout_type::column_major`` fixes the strided index scheme and disable ``resize`` and constructor overloads taking a set of strides or a layout value as parameter. The default value of the template parameter is ``XTENSOR_DEFAULT_LAYOUT``.
- at runtime if the previous template parameter was set to ``layout_type::dynamic``. In that case, ``resize`` and constructor overloads allow specifying a set of strides or a layout value to avoid strides computation. If neither strides nor layout is specified when instantiating or resizing a multi-dimensional array, strides corresponding to ``XTENSOR_DEFAULT_LAYOUT`` are used.

The following example shows how to initialize a multi-dimensional array of dynamic layout with specified strides:

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"

    std::vector<size_t> shape = { 3, 2, 4 };
    std::vector<size_t> strides = { 8, 4, 1 };
    xt::xarray<double, xt::layout_type::dynamic> a(shape, strides);

However, this requires to carefully compute the strides to avoid buffer overflow when accessing elements of the array. We can use the following shortcut to specify the strides instead of computing them:

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"

    std::vector<size_t> shape = { 3, 2, 4 };
    xt::xarray<double, xt::layout_type::dynamic> a(shape, xt::layout_type::row_major);

If the layout of the array can be fixed at compile time, we can make it even simpler:

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"

    std::vector<size_t> shape = { 3, 2, 4 };
    xt::xarray<double, xt::layout_type::row_major> a(shape);
    // this shortcut is equivalent:
    // xt::xarray<double> a(shape);

However, in the latter case, the layout of the array is forced to ``row_major`` at compile time, and therefore cannot be changed at runtime.

Runtime vs Compile-time dimensionality
--------------------------------------

Three container classes implementing multidimensional arrays are provided: ``xarray`` and ``xtensor`` and ``xtensor_fixed``.

- ``xarray`` can be reshaped dynamically to any number of dimensions. It is the container that is the most similar to numpy arrays.
- ``xtensor`` has a dimension set at compilation time, which enables many optimizations. For example, shapes and strides
  of ``xtensor`` instances are allocated on the stack instead of the heap.
- ``xtensor_fixed`` has a shape fixed at compile time. This allows even more optimizations, such as allocating the storage for the container
  on the stack, as well as computing strides and backstrides at compile time, making the allocation of this container extremely cheap.

Let's use ``xtensor`` instead of ``xarray`` in the previous example:

.. code::

    #include <array>
    #include "xtensor/xtensor.hpp"

    std::array<size_t, 3> shape = { 3, 2, 4 };
    xt::xtensor<double, 3> a(shape);
    // this is equivalent to
    // xt::xtensor<double, 3, xt::layout_type::row_major> a(shape);

Or when using ``xtensor_fixed``:

.. code::

    #include "xtensor/xfixed.hpp"

    xt::xtensor_fixed<double, xt::xshape<3, 2, 4>> a();
    // or xt::xtensor_fixed<double, xt::xshape<3, 2, 4>, xt::layout_type::row_major>()

``xarray``, ``xtensor`` and ``xtensor_fixed`` containers are all ``xexpression`` s and can be involved and mixed in mathematical expressions, assigned to each
other etc... They provide an augmented interface compared to other ``xexpression`` types:

- Each method exposed in ``xexpression`` interface has its non-const counterpart exposed by ``xarray``, ``xtensor`` and ``xtensor_fixed``.
- ``reshape()`` reshapes the container in place, and the global size of the container has to stay the same.
- ``resize()`` resizes the container in place, that is, if the global size of the container doesn't change, no memory allocation occurs.
- ``strides()`` returns the strides of the container, used to compute the position of an element in the underlying buffer.

Reshape
-------

The ``reshape`` method accepts any kind of 1D-container, you don't have to pass an instance of ``shape_type``. It only requires the new shape to be
compatible with the old one, that is, the number of elements in the container must remain the same:

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<int> a = { 1, 2, 3, 4, 5, 6, 7, 8};
    // The following two lines ...
    std::array<std::size_t, 2> sh1 = {2, 4};
    a.reshape(sh1);
    // ... are equivalent to the following two lines ...
    xt::xarray<int>::shape_type sh2({2, 4});
    a.reshape(sh2);
    // ... which are equivalent to the following
    a.reshape({2, 4});

One of the values in the ``shape`` argument can be -1. In this case, the value is inferred from the number of elements in the container and the remaining
values in the ``shape``:

.. code::

    #include "xtensor/xarray.hpp"
    xt::xarray<int> a = { 1, 2, 3, 4, 5, 6, 7, 8};
    a.reshape({2, -1});
    // a.shape() return {2, 4}

Performance
-----------

The dynamic dimensionality of ``xarray`` comes at a cost. Since the dimension is unknown at build time, the sequences holding shape and strides of ``xarray`` instances are heap-allocated, which makes it significantly more expensive than ``xtensor``. Shape and strides of ``xtensor`` are stack-allocated which makes them more efficient.

More generally, the library implements a ``promote_shape`` mechanism at build time to determine the optimal sequence type to hold the shape of an expression. The shape type of a broadcasting expression whose members have a dimensionality determined at compile time will have a stack-allocated shape. If a single member of a broadcasting expression has a dynamic dimension (for example an ``xarray``), it bubbles up to the entire broadcasting expression which will have a heap-allocated shape. The same hold for views, broadcast expressions, etc...

Aliasing and temporaries
------------------------

In some cases, an expression should not be directly assigned to a container. Instead, it has to be assigned to a temporary variable before being copied
into the destination container. A typical case where this happens is when the destination container is involved in the expression and has to be resized.
This phenomenon is known as *aliasing*.

To prevent this, `xtensor` assigns the expression to a temporary variable before copying it. In the case of ``xarray``, this results in an extra dynamic memory
allocation and copy.

However, if the left-hand side is not involved in the expression being assigned, no temporary variable should be required. `xtensor` cannot detect such cases
automatically and applies the "temporary variable rule" by default. A mechanism is provided to forcibly prevent usage of a temporary variable:

.. code::

    #include "xtensor/xarray.hpp"
    #include "xtensor/xnoalias.hpp"

    // a, b, and c are xt::xarrays previously initialized
    xt::noalias(b) = a + c;
    // Even if b has to be resized, a+c will be assigned directly to it
    // No temporary variable will be involved

Example of aliasing
~~~~~~~~~~~~~~~~~~~

The aliasing phenomenon is illustrated in the following example:

.. code::

    #include <vector>
    #include "xtensor/xarray.hpp"

    std::vector<size_t> a_shape = {3, 2, 4};
    xt::xarray<double> a(a_shape);

    std::vector<size_t> b_shape = {2, 4};
    xt::xarray<double> b(b_shape);

    b = a + b;
    // b appears on both left-hand and right-hand sides of the statement

In the above example, the shape of ``a + b`` is ``{ 3, 2, 4 }``. Therefore, ``b`` must first be resized, which impacts how the right-hand side is computed.

If the values of ``b`` were copied into the new buffer directly without an intermediary variable, then we would have
``new_b(0, i, j) == old_b(i, j) for (i,j) in [0,1] x [0, 3]``. After the resize of ``bb``, ``a(0, i, j) + b(0, i, j)`` is assigned to ``b(0, i, j)``, then,
due to broadcasting rules, ``a(1, i, j) + b(0, i, j)`` is assigned to ``b(1, i, j)``. The issue is ``b(0, i, j)`` has been changed by the previous assignment.
