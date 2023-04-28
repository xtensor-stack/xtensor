.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

From NumPy to xtensor
=====================

.. image:: numpy.svg
   :height: 100px
   :align: right

.. raw:: html

   <style>
   .rst-content table.docutils {
       width: 100%;
       table-layout: fixed;
       border: none;
   }

   table.docutils th {
       text-align: center;
   }

   table.docutils .line-block {
       margin-left: 0;
       margin-bottom: 0;
   }

   table.docutils code.literal {
       color: initial;
   }

   code.docutils {
       background: initial;
       border: none;
   }

   * {
       border: none;
   }

   .rst-content table.docutils thead {
       background-color: #d0e0e0;
   }

   .rst-content table.docutils td {
       border-bottom: none;
       border-left: none;
   }

   .rst-content table.docutils td > p {
       overflow: auto;
   }

   .rst-content table.docutils tr:hover {
       background-color: #d0e0e0;
   }

   .rst-content table.docutils:not(.field-list) tr:nth-child(2n-1):hover td {
       background-color: initial;
   }

   #linear-algebra table.docutils thead .row-odd {
       background: #ffdddd;
   }

   #linear-algebra tr:nth-child(2n-1) td {
       background: #f9f3f3;
   }

   #linear-algebra tr:hover {
       background: #ffdddd;
   }

   #linear-algebra tr:nth-child(2n-1):hover td {
       background-color: initial;
   }
   </style>

Containers
----------

Two container types are provided. :cpp:type:`xt::xarray` (dynamic number of dimensions)
and :cpp:type:`xt::xtensor` (static number of dimensions).

.. table::
   :widths: 50 50

   +------------------------------------------------------+------------------------------------------------------------------------+
   |             Python 3 - NumPy                         |               C++ 14 - xtensor                                         |
   +======================================================+========================================================================+
   | :any:`np.array([[3, 4], [5, 6]]) <numpy.array>`      || :cpp:type:`xt::xarray\<double\>({{3, 4}, {5, 6}}) <xt::xarray>`       |
   |                                                      || :cpp:type:`xt::xtensor\<double, 2\>({{3, 4}, {5, 6}}) <xt::xtensor>`  |
   +------------------------------------------------------+------------------------------------------------------------------------+
   | :any:`arr.reshape([3, 4]) <numpy.ndarray.reshape>`   | :cpp:func:`arr.reshape({3, 4}) <xt::xstrided_container::reshape>`      |
   +------------------------------------------------------+------------------------------------------------------------------------+
   | :any:`arr.astype(np.float64) <numpy.ndarray.astype>` | :cpp:func:`xt::cast\<double\>(arr) <xt::cast>`                         |
   +------------------------------------------------------+------------------------------------------------------------------------+

Initializers
------------

Lazy helper functions return tensor expressions. Return types don't hold any value and are
evaluated upon access or assignment. They can be assigned to a container or directly used in
expressions.

.. table::
   :widths: 50 50

   +----------------------------------------------------------------+-------------------------------------------------------------------+
   |             Python 3 - NumPy                                   |               C++ 14 - xtensor                                    |
   +================================================================+===================================================================+
   | :any:`np.linspace(1.0, 10.0, 100) <numpy.linspace>`            | :cpp:func:`xt::linspace\<double\>(1.0, 10.0, 100) <xt::linspace>` |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.logspace(2.0, 3.0, 4) <numpy.logspace>`               | :cpp:func:`xt::logspace\<double\>(2.0, 3.0, 4) <xt::logspace>`    |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.arange(3, 7) <numpy.arange>`                          | :cpp:func:`xt::arange(3, 7) <xt::arange>`                         |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.eye(4) <numpy.eye>`                                   | :cpp:func:`xt::eye(4) <xt::eye>`                                  |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.zeros([3, 4]) <numpy.zeros>`                          | :cpp:func:`xt::zeros\<double\>({3, 4}) <xt::zeros>`               |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.ones([3, 4]) <numpy.ones>`                            | :cpp:func:`xt::ones\<double\>({3, 4}) <xt::ones>`                 |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.empty([3, 4]) <numpy.empty>`                          | :cpp:func:`xt::empty\<double\>({3, 4}) <xt::empty>`               |
   +----------------------------------------------------------------+-------------------------------------------------------------------+
   | :any:`np.meshgrid(x0, x1, x2, indexing='ij') <numpy.meshgrid>` | :cpp:func:`xt::meshgrid(x0, x1, x2) <xt::meshgrid>`               |
   +----------------------------------------------------------------+-------------------------------------------------------------------+

xtensor's :cpp:func:`meshgrid <xt::meshgrid>` implementation corresponds to numpy's ``'ij'`` indexing order.

Slicing and indexing
--------------------

See :any:`numpy indexing <numpy:arrays.indexing>` page.

.. table::
   :widths: 50 50

   +-----------------------------------------+---------------------------------------------------------------------------+
   |             Python 3 - NumPy            |                   C++ 14 - xtensor                                        |
   +=========================================+===========================================================================+
   | ``a[3, 2]``                             | :cpp:func:`a(3, 2) <xt::xcontainer::operator()>`                          |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | :any:`a.flat[4] <numpy.ndarray.flat>`   | :cpp:func:`a.flat(4) <xt::xcontainer::flat>`                              |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | ``a[3]``                                || :cpp:func:`xt::view(a, 3, xt::all()) <xt::view>`                         |
   |                                         || :cpp:func:`xt::row(a, 3) <xt::row>`                                      |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | ``a[:, 2]``                             || :cpp:func:`xt::view(a, xt::all(), 2) <xt::view>`                         |
   |                                         || :cpp:func:`xt::col(a, 2) <xt::col>`                                      |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | ``a[:5, 1:]``                           | :cpp:func:`xt::view(a, xt::range(_, 5), xt::range(1, _)) <xt::range>`     |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | ``a[5:1:-1, :]``                        | :cpp:func:`xt::view(a, xt::range(5, 1, -1), xt::all()) <xt::all>`         |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | ``a[..., 3]``                           | :cpp:func:`xt::strided_view(a, {xt::ellipsis(), 3}) <xt::ellipsis>`       |
   +-----------------------------------------+---------------------------------------------------------------------------+
   | :any:`a[:, np.newaxis] <numpy.newaxis>` | :cpp:func:`xt::view(a, xt::all(), xt::newaxis()) <xt::newaxis>`           |
   +-----------------------------------------+---------------------------------------------------------------------------+

Broadcasting
------------

xtensor offers lazy numpy-style broadcasting, and universal functions. Unlike numpy, no copy
or temporary variables are created.

.. table::
   :widths: 50 50

   +-----------------------------------------------------+------------------------------------------------------------------+
   |             Python 3 - NumPy                        |                   C++ 14 - xtensor                               |
   +=====================================================+==================================================================+
   | :any:`np.broadcast(a, [4, 5, 7]) <numpy.broadcast>` | :cpp:func:`xt::broadcast(a, {4, 5, 7}) <xt::broadcast>`          |
   +-----------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.vectorize(f) <numpy.vectorize>`            | :cpp:func:`xt::vectorize(f) <xt::vectorize>`                     |
   +-----------------------------------------------------+------------------------------------------------------------------+
   | ``a[a > 5]``                                        | :cpp:func:`xt::filter(a, a > 5) <xt::filter>`                    |
   +-----------------------------------------------------+------------------------------------------------------------------+
   | ``a[[0, 1], [0, 0]]``                               | :cpp:func:`xt::index_view(a, {{0, 0}, {1, 0}}) <xt::index_view>` |
   +-----------------------------------------------------+------------------------------------------------------------------+

Random
------

The random module provides simple ways to create random tensor expressions, lazily.
See :any:`numpy.random` and :ref:`xtensor random <random>` page.

.. table::
   :widths: 50 50

   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   |            Python 3 - NumPy                                           |                C++ 14 - xtensor                                                   |
   +=======================================================================+===================================================================================+
   | :any:`np.random.seed(0) <numpy.random.seed>`                          | :cpp:func:`xt::random::seed(0) <xt::random::seed>`                                |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.randn(10, 10) <numpy.random.randn>`                   | :cpp:func:`xt::random::randn\<double\>({10, 10}) <xt::random::randn>`             |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.randint(10, 10) <numpy.random.randint>`               | :cpp:func:`xt::random::randint\<int\>({10, 10}) <xt::random::randint>`            |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.rand(3, 4) <numpy.random.rand>`                       | :cpp:func:`xt::random::rand\<double\>({3, 4}) <xt::random::rand>`                 |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.choice(arr, 5[, replace][, p]) <numpy.random.choice>` | :cpp:func:`xt::random::choice(arr, 5[, weights][, replace]) <xt::random::choice>` |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.shuffle(arr) <numpy.random.shuffle>`                  | :cpp:func:`xt::random::shuffle(arr) <xt::random::shuffle>`                        |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
   | :any:`np.random.permutation(30) <numpy.random.permutation>`           | :cpp:func:`xt::random::permutation(30) <xt::random::permutation>`                 |
   +-----------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Concatenation, splitting, squeezing
-----------------------------------

Concatenating expressions does not allocate memory, it returns a tensor or view expression holding
closures on the specified arguments.

.. table::
   :widths: 50 50

   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   |            Python 3 - NumPy                                                 |                C++ 14 - xtensor                                            |
   +=============================================================================+============================================================================+
   | :any:`np.stack([a, b, c], axis=1) <numpy.stack>`                            | :cpp:func:`xt::stack(xtuple(a, b, c), 1) <xt::stack>`                      |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.hstack([a, b, c]) <numpy.hstack>`                                  | :cpp:func:`xt::hstack(xtuple(a, b, c)) <xt::hstack>`                       |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.vstack([a, b, c]) <numpy.vstack>`                                  | :cpp:func:`xt::vstack(xtuple(a, b, c)) <xt::vstack>`                       |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.concatenate([a, b, c], axis=1) <numpy.concatenate>`                | :cpp:func:`xt::concatenate(xtuple(a, b, c), 1) <xt::concatenate>`          |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.tile(a, reps) <numpy.tile>`                                        | :cpp:func:`xt::tile(a, reps) <xt::tile>`                                   |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.squeeze(a) <numpy.squeeze>`                                        | :cpp:func:`xt::squeeze(a) <xt::squeeze>`                                   |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.expand_dims(a, 1) <numpy.expand_dims>`                             | :cpp:func:`xt::expand_dims(a ,1) <xt::expand_dims>`                        |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.atleast_3d(a) <numpy.atleast_3d>`                                  | :cpp:func:`xt::atleast_3d(a) <xt::atleast_3d>`                             |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.split(a, 4, axis=0) <numpy.split>`                                 | :cpp:func:`xt::split(a, 4, 0) <xt::split>`                                 |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.hsplit(a, 4) <numpy.hsplit>`                                       | :cpp:func:`xt::hsplit(a, 4) <xt::hsplit>`                                  |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.vsplit(a, 4) <numpy.vsplit>`                                       | :cpp:func:`xt::vsplit(a, 4) <xt::vsplit>`                                  |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.trim_zeros(a, trim='fb') <numpy.trim_zeros>`                       | :cpp:func:`xt::trim_zeros(a, "fb") <xt::trim_zeros>`                       |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+
   | :any:`np.pad(a, pad_width, mode='constant', constant_values=0) <numpy.pad>` | :cpp:func:`xt::pad(a, pad_width[, xt::pad_mode::constant][, 0]) <xt::pad>` |
   +-----------------------------------------------------------------------------+----------------------------------------------------------------------------+

Rearrange elements
------------------

In the same spirit as concatenation, the following operations do not allocate any memory and do
not modify the underlying xexpression.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Python3 - NumPy
     - C++14 - xtensor
   * - :any:`np.nan_to_num(a) <numpy.nan_to_num>`
     - :cpp:func:`xt::nan_to_num(a) <xt::nan_to_num>`
   * - :any:`np.diag(a) <numpy.diag>`
     - :cpp:func:`xt::diag(a) <xt::diag>`
   * - :any:`np.diagonal(a) <numpy.diagonal>`
     - :cpp:func:`xt::diagonal(a) <xt::diagonal>`
   * - :any:`np.triu(a) <numpy.triu>`
     - :cpp:func:`xt::triu(a) <xt::triu>`
   * - :any:`np.tril(a, k=1) <numpy.tril>`
     - :cpp:func:`xt::tril(a, 1) <xt::tril>`
   * - :any:`np.flip(a, axis=3) <numpy.flip>`
     - :cpp:func:`xt::flip(a, 3) <xt::flip>`
   * - :any:`np.flipud(a) <numpy.flipud>`
     - :cpp:func:`xt::flip(a, 0) <xt::flip>`
   * - :any:`np.fliplr(a) <numpy.fliplr>`
     - :cpp:func:`xt::flip(a, 1) <xt::flip>`
   * - :any:`np.transpose(a, (1, 0, 2)) <numpy.transpose>`
     - :cpp:func:`xt::transpose(a, {1, 0, 2}) <xt::transpose>`
   * - :any:`np.swapaxes(a, 0, -1) <numpy.swapaxes>`
     - :cpp:func:`xt::swapaxes(a, 0, -1) <xt::swapaxes>`
   * - :any:`np.moveaxis(a, 0, -1) <numpy.moveaxis>`
     - :cpp:func:`xt::moveaxis(a, 0, -1) <xt::moveaxis>`
   * - :any:`np.ravel(a, order='F') <numpy.ravel>`
     - :cpp:func:`xt::ravel\<xt::layout_type::column_major\>(a) <xt::ravel>`
   * - :any:`np.rot90(a) <numpy.rot90>`
     - :cpp:func:`xt::rot90(a) <xt::rot90>`
   * - :any:`np.rot90(a, 2, (1, 2)) <numpy.rot90>`
     - :cpp:func:`xt::rot90\<2\>(a, {1, 2}) <xt::rot90>`
   * - :any:`np.roll(a, 2, axis=1) <numpy.roll>`
     - :cpp:func:`xt::roll(a, 2, 1) <xt::roll>`

Iteration
---------

xtensor follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in
different fashions.

.. table::
   :widths: 50 50

   +-----------------------------------------------------------+------------------------------------------------+
   |            Python 3 - NumPy                               |                C++ 14 - xtensor                |
   +===========================================================+================================================+
   | :any:`for x in np.nditer(a): <numpy.nditer>`              |  ``for(auto it=a.begin(); it!=a.end(); ++it)`` |
   +-----------------------------------------------------------+------------------------------------------------+
   | Iterating over ``a`` with a prescribed broadcasting shape | | ``a.begin({3, 4})``                          |
   |                                                           | | ``a.end({3, 4})``                            |
   +-----------------------------------------------------------+------------------------------------------------+
   | Iterating over ``a`` in a row-major fashion               | | ``a.begin<xt::layout_type::row_major>()``    |
   |                                                           | | ``a.begin<xt::layout_type::row_major>()``    |
   +-----------------------------------------------------------+------------------------------------------------+
   | Iterating over ``a`` in a column-major fashion            | | ``a.begin<xt::layout_type::column_major>()`` |
   |                                                           | | ``a.end<xt::layout_type::column_major>()``   |
   +-----------------------------------------------------------+------------------------------------------------+

Logical
-------

Logical universal functions are truly lazy.
:cpp:func:`xt::where(condition, a, b) <xt::where>` does not evaluate ``a`` where ``condition``
is falsy, and it does not evaluate ``b`` where ``condition`` is truthy.

.. table::
   :widths: 50 50

   +-------------------------------------------------+------------------------------------------------+
   |            Python 3 - NumPy                     |                C++ 14 - xtensor                |
   +=================================================+================================================+
   | :any:`np.where(a > 5, a, b) <numpy.where>`      | :cpp:func:`xt::where(a > 5, a, b) <xt::where>` |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.where(a > 5) <numpy.where>`            | :cpp:func:`xt::where(a > 5) <xt::where>`       |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.argwhere(a > 5) <numpy.argwhere>`      | :cpp:func:`xt::argwhere(a > 5) <xt::argwhere>` |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.any(a) <numpy.any>`                    | :cpp:func:`xt::any(a) <xt::any>`               |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.all(a) <numpy.all>`                    | :cpp:func:`xt::all(a) <xt::all>`               |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.isin(a, b) <numpy.isin>`               | :cpp:func:`xt::isin(a, b) <xt::isin>`          |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.in1d(a, b) <numpy.in1d>`               | :cpp:func:`xt::in1d(a, b) <xt::in1d>`          |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.logical_and(a, b) <numpy.logical_and>` | ``a && b``                                     |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.logical_or(a, b) <numpy.logical_or>`   | ``a || b``                                     |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.isclose(a, b) <numpy.isclose>`         | :cpp:func:`xt::isclose(a, b) <xt::isclose>`    |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`np.allclose(a, b) <numpy.allclose>`       | :cpp:func:`xt::allclose(a, b) <xt::allclose>`  |
   +-------------------------------------------------+------------------------------------------------+
   | :any:`a = ~b <numpy.invert>`                    | ``a = !b``                                     |
   +-------------------------------------------------+------------------------------------------------+

Indices
-------

.. table::
   :widths: 50 50

   +-------------------------------------------------------------------------+-----------------------------------------------------------------------+
   |            Python 3 - NumPy                                             |                C++ 14 - xtensor                                       |
   +=========================================================================+=======================================================================+
   | :any:`np.ravel_multi_index(indices, a.shape) <numpy.ravel_multi_index>` | :cpp:func:`xt::ravel_indices(indices, a.shape()) <xt::ravel_indices>` |
   +-------------------------------------------------------------------------+-----------------------------------------------------------------------+

Comparisons
-----------

.. table::
   :widths: 50 50

   +-----------------------------------------------------+----------------------------------------------------------+
   |            Python 3 - NumPy                         |                C++ 14 - xtensor                          |
   +=====================================================+==========================================================+
   | :any:`np.equal(a, b) <numpy.equal>`                 | :cpp:func:`xt::equal(a, b) <xt::equal>`                  |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.not_equal(a, b) <numpy.not_equal>`         | :cpp:func:`xt::not_equal(a, b) <xt::not_equal>`          |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.less(a, b) <numpy.less>`                   || :cpp:func:`xt::less(a, b) <xt::less>`                   |
   |                                                     || ``a < b``                                               |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.less_equal(a, b) <numpy.less_equal>`       || :cpp:func:`xt::less_equal(a, b) <xt::less_equal>`       |
   |                                                     || ``a <= b``                                              |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.greater(a, b) <numpy.greater>`             || :cpp:func:`xt::greater(a, b) <xt::greater>`             |
   |                                                     || ``a > b``                                               |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.greater_equal(a, b) <numpy.greater_equal>` || :cpp:func:`xt::greater_equal(a, b) <xt::greater_equal>` |
   |                                                     || ``a >= b``                                              |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.nonzero(a) <numpy.nonzero>`                | :cpp:func:`xt::nonzero(a) <xt::nonzero>`                 |
   +-----------------------------------------------------+----------------------------------------------------------+
   | :any:`np.flatnonzero(a) <numpy.flatnonzero>`        | :cpp:func:`xt::flatnonzero(a) <xt::flatnonzero>`         |
   +-----------------------------------------------------+----------------------------------------------------------+

Minimum, Maximum, Sorting
-------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Python3 - NumPy
     - C++14 - xtensor
   * - :any:`np.amin(a) <numpy.amin>`
     - :cpp:func:`xt::amin(a) <xt::amin>`
   * - :any:`np.amax(a) <numpy.amax>`
     - :cpp:func:`xt::amax(a) <xt::amax>`
   * - :any:`np.argmin(a) <numpy.argmin>`
     - :cpp:func:`xt::argmin(a) <xt::argmin>`
   * - :any:`np.argmax(a, axis=1) <numpy.argmax>`
     - :cpp:func:`xt::argmax(a, 1) <xt::argmax>`
   * - :any:`np.sort(a, axis=1) <numpy.sort>`
     - :cpp:func:`xt::sort(a, 1) <xt::sort>`
   * - :any:`np.argsort(a, axis=1) <numpy.argsort>`
     - :cpp:func:`xt::argsort(a, 1) <xt::argsort>`
   * - :any:`np.unique(a) <numpy.unique>`
     - :cpp:func:`xt::unique(a) <xt::unique>`
   * - :any:`np.setdiff1d(ar1, ar2) <numpy.setdiff1d>`
     - :cpp:func:`xt::setdiff1d(ar1, ar2) <xt::setdiff1d>`
   * - :any:`np.partition(a, kth) <numpy.partition>`
     - :cpp:func:`xt::partition(a, kth) <xt::partition>`
   * - :any:`np.argpartition(a, kth) <numpy.argpartition>`
     - :cpp:func:`xt::argpartition(a, kth) <xt::argpartition>`
   * - :any:`np.quantile(a, [.1 .3], method="linear") <numpy.quantile>`
     - :cpp:func:`xt::quantile(a, {.1, .3}, xt::quantile_method::linear) <xt::quantile>`
   * - :any:`np.quantile(a, [.1, .3], axis=1 method="linear") <numpy.quantile>`
     - :cpp:func:`xt::quantile(a, {.1, .3}, 1, xt::quantile_method::linear) <xt::quantile>`
   * -
     - :cpp:func:`xt::quantile(a, {.1, .3}, 1, 1.0, 1.0) <xt::quantile>`
   * - :any:`np.median(a, axis=1) <numpy.median>`
     - :cpp:func:`xt::median(a, 1) <xt::median>`

Complex numbers
---------------

Functions :cpp:func:`xt::real` and :cpp:func:`xt::imag` respectively return views on the real and imaginary part
of a complex expression.
The returned value is an expression holding a closure on the passed argument.

.. table::
   :widths: 50 50

   +--------------------------------+------------------------------------+
   |            Python 3 - NumPy    |                C++ 14 - xtensor    |
   +================================+====================================+
   | :any:`np.real(a) <numpy.real>` | :cpp:func:`xt::real(a) <xt::real>` |
   +--------------------------------+------------------------------------+
   | :any:`np.imag(a) <numpy.imag>` | :cpp:func:`xt::imag(a) <xt::imag>` |
   +--------------------------------+------------------------------------+
   | :any:`np.conj(a) <numpy.conj>` | :cpp:func:`xt::conj(a) <xt::conj>` |
   +--------------------------------+------------------------------------+

- The constness and value category (rvalue / lvalue) of :cpp:func:`xt::real(a) <xt::real>` is the same as that of ``a``.
  Hence, if ``a`` is a non-const lvalue, :cpp:func:`real(a) <xt::real>` is an non-const lvalue reference, to which
  one can assign a real expression.
- If ``a`` has complex values, the same holds for :cpp:func:`xt::imag(a) <xt::imag>`. The constness and value category of
  :cpp:func:`xt::imag(a) <xt::imag>` is the same as that of ``a``.
- If ``a`` has real values, :cpp:func:`xt::imag(a) <xt::imag>` returns :cpp:func:`xt::zeros(a.shape()) <xt::zeros>`.

Reducers
--------

Reducers accumulate values of tensor expressions along specified axes. When no axis is specified,
values are accumulated along all axes. Reducers are lazy, meaning that returned expressions don't
hold any values and are computed upon access or assignment.

.. table::
   :widths: 50 50

   +---------------------------------------------------------------+--------------------------------------------------------------+
   |            Python 3 - NumPy                                   |                C++ 14 - xtensor                              |
   +===============================================================+==============================================================+
   | :any:`np.sum(a, axis=(0, 1)) <numpy.sum>`                     | :cpp:func:`xt::sum(a, {0, 1}) <xt::sum>`                     |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.sum(a, axis=1) <numpy.sum>`                          | :cpp:func:`xt::sum(a, 1) <xt::sum>`                          |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.sum(a) <numpy.sum>`                                  | :cpp:func:`xt::sum(a) <xt::sum>`                             |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.prod(a, axis=(0, 1)) <numpy.prod>`                   | :cpp:func:`xt::prod(a, {0, 1}) <xt::prod>`                   |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.prod(a, axis=1) <numpy.prod>`                        | :cpp:func:`xt::prod(a, 1) <xt::prod>`                        |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.prod(a) <numpy.prod>`                                | :cpp:func:`xt::prod(a) <xt::prod>`                           |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.mean(a, axis=(0, 1)) <numpy.mean>`                   | :cpp:func:`xt::mean(a, {0, 1}) <xt::mean>`                   |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.mean(a, axis=1) <numpy.mean>`                        | :cpp:func:`xt::mean(a, 1) <xt::mean>`                        |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.mean(a) <numpy.mean>`                                | :cpp:func:`xt::mean(a) <xt::mean>`                           |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.std(a, [axis]) <numpy.std>`                          | :cpp:func:`xt::stddev(a, [axis]) <xt::stddev>`               |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.var(a, [axis]) <numpy.var>`                          | :cpp:func:`xt::variance(a, [axis]) <xt::variance>`           |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.diff(a[, n, axis]) <numpy.diff>`                     | :cpp:func:`xt::diff(a[, n, axis]) <xt::diff>`                |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.trapz(a, dx=2.0, axis=-1) <numpy.trapz>`             | :cpp:func:`xt::trapz(a, 2.0, -1) <xt::trapz>`                |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.trapz(a, x=b, axis=-1) <numpy.trapz>`                | :cpp:func:`xt::trapz(a, b, -1) <xt::trapz>`                  |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.count_nonzero(a, axis=(0, 1)) <numpy.count_nonzero>` | :cpp:func:`xt::count_nonzero(a, {0, 1}) <xt::count_nonzero>` |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.count_nonzero(a, axis=1) <numpy.count_nonzero>`      | :cpp:func:`xt::count_nonzero(a, 1) <xt::count_nonzero>`      |
   +---------------------------------------------------------------+--------------------------------------------------------------+
   | :any:`np.count_nonzero(a) <numpy.count_nonzero>`              | :cpp:func:`xt::count_nonzero(a) <xt::count_nonzero>`         |
   +---------------------------------------------------------------+--------------------------------------------------------------+

More generally, one can use the :cpp:func:`xt::reduce(function, input, axes) <xt::reduce>` which allows the specification
of an arbitrary binary function for the reduction.
The binary function must be commutative and associative up to rounding errors.

NaN functions
-------------

NaN functions allow disregarding NaNs during computation, changing the effective number of elements
considered in reductions.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Python3 - NumPy
     - C++14 - xtensor
   * - :any:`np.nan_to_num(a) <numpy.nan_to_num>`
     - :cpp:func:`xt::nan_to_num(a) <xt::nan_to_num>`
   * - :any:`np.nanmin(a) <numpy.nanmin>`
     - :cpp:func:`xt::nanmin(a) <xt::nanmin>`
   * - :any:`np.nanmin(a, axis=(0, 1)) <numpy.nanmin>`
     - :cpp:func:`xt::nanmin(a, {0, 1}) <xt::nanmin>`
   * - :any:`np.nanmax(a) <numpy.nanmax>`
     - :cpp:func:`xt::nanmax(a) <xt::nanmax>`
   * - :any:`np.nanmax(a, axis=(0, 1)) <numpy.nanmax>`
     - :cpp:func:`xt::nanmax(a, {0, 1}) <xt::nanmax>`
   * - :any:`np.nansum(a) <numpy.nansum>`
     - :cpp:func:`xt::nansum(a) <xt::nansum>`
   * - :any:`np.nansum(a, axis=0) <numpy.nansum>`
     - :cpp:func:`xt::nansum(a, 0) <xt::nansum>`
   * - :any:`np.nansum(a, axis=(0, 1)) <numpy.nansum>`
     - :cpp:func:`xt::nansum(a, {0, 1}) <xt::nansum>`
   * - :any:`np.nanprod(a) <numpy.nanprod>`
     - :cpp:func:`xt::nanprod(a) <xt::nanprod>`
   * - :any:`np.nanprod(a, axis=0) <numpy.nanprod>`
     - :cpp:func:`xt::nanprod(a, 0) <xt::nanprod>`
   * - :any:`np.nanprod(a, axis=(0, 1)) <numpy.nanprod>`
     - :cpp:func:`xt::nanprod(a, {0, 1}) <xt::nanprod>`
   * - :any:`np.nancumsum(a) <numpy.nancumsum>`
     - :cpp:func:`xt::nancumsum(a) <xt::nancumsum>`
   * - :any:`np.nancumsum(a, axis=0) <numpy.nancumsum>`
     - :cpp:func:`xt::nancumsum(a, 0) <xt::nancumsum>`
   * - :any:`np.nancumprod(a) <numpy.nancumsum>`
     - :cpp:func:`xt::nancumsum(a) <xt::nancumsum>`
   * - :any:`np.nancumprod(a, axis=0) <numpy.nancumsum>`
     - :cpp:func:`xt::nancumsum(a, 0) <xt::nancumsum>`
   * - :any:`np.nanmean(a) <numpy.nanmean>`
     - :cpp:func:`xt::nanmean(a) <xt::nanmean>`
   * - :any:`np.nanmean(a, axis=(0, 1)) <numpy.nanmean>`
     - :cpp:func:`xt::nanmean(a, {0, 1}) <xt::nanmean>`
   * - :any:`np.nanvar(a) <numpy.nanvar>`
     - :cpp:func:`xt::nanvar(a) <xt::nanvar>`
   * - :any:`np.nanvar(a, axis=(0, 1)) <numpy.nanvar>`
     - :cpp:func:`xt::nanvar(a, {0, 1}) <xt::nanvar>`
   * - :any:`np.nanstd(a) <numpy.nanstd>`
     - :cpp:func:`xt::nanstd(a) <xt::nanstd>`
   * - :any:`np.nanstd(a, axis=(0, 1)) <numpy.nanstd>`
     - :cpp:func:`xt::nanstd(a, {0, 1}) <xt::nanstd>`

I/O
---

**Print options**

These options determine the way floating point numbers, tensors and other xtensor expressions are displayed.

.. table::
   :widths: 50 50

   +--------------------------------------------------------------------+----------------------------------------------------------------------------------------+
   |            Python 3 - NumPy                                        |                C++ 14 - xtensor                                                        |
   +====================================================================+========================================================================================+
   | :any:`np.set_printoptions(precision=4) <numpy.set_printoptions>`   | :cpp:func:`xt::print_options::set_precision(4) <xt::print_options::set_precision>`     |
   +--------------------------------------------------------------------+----------------------------------------------------------------------------------------+
   | :any:`np.set_printoptions(threshold=5) <numpy.set_printoptions>`   | :cpp:func:`xt::print_options::set_threshold(5) <xt::print_options::set_threshold>`     |
   +--------------------------------------------------------------------+----------------------------------------------------------------------------------------+
   | :any:`np.set_printoptions(edgeitems=3) <numpy.set_printoptions>`   | :cpp:func:`xt::print_options::set_edgeitems(3) <xt::print_options::set_edgeitems>`     |
   +--------------------------------------------------------------------+----------------------------------------------------------------------------------------+
   | :any:`np.set_printoptions(linewidth=100) <numpy.set_printoptions>` | :cpp:func:`xt::print_options::set_line_width(100) <xt::print_options::set_line_width>` |
   +--------------------------------------------------------------------+----------------------------------------------------------------------------------------+

**Reading npy, csv file formats**

Functions :cpp:func:`xt::load_csv` and :cpp:func:`xt::dump_csv` respectively take input and output streams as arguments.

.. table::
   :widths: 50 50

   +------------------------------------------------------------+-------------------------------------------------------------+
   |            Python 3 - NumPy                                |                C++ 14 - xtensor                             |
   +============================================================+=============================================================+
   | :any:`np.load(filename) <numpy.load>`                      | :cpp:func:`xt::load_npy\<double\>(filename) <xt::load_npy>` |
   +------------------------------------------------------------+-------------------------------------------------------------+
   | :any:`np.save(filename, arr) <numpy.save>`                 | :cpp:func:`xt::dump_npy(filename, arr) <xt::dump_npy>`      |
   +------------------------------------------------------------+-------------------------------------------------------------+
   | :any:`np.loadtxt(filename, delimiter=',') <numpy.loadtxt>` | :cpp:func:`xt::load_csv\<double\>(stream) <xt::load_csv>`   |
   +------------------------------------------------------------+-------------------------------------------------------------+

Mathematical functions
----------------------

xtensor universal functions are provided for a large set number of mathematical functions.

**Basic functions:**

.. table::
   :widths: 50 50

   +------------------------------------------------------------+----------------------------------------------------------------+
   |            Python 3 - NumPy                                |                C++ 14 - xtensor                                |
   +============================================================+================================================================+
   | :any:`np.absolute(a) <numpy.absolute>`                     | :cpp:func:`xt::abs(a) <xt::abs>`                               |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.sign(a) <numpy.sign>`                             | :cpp:func:`xt::sign(a) <xt::sign>`                             |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.remainder(a, b) <numpy.remainder>`                | :cpp:func:`xt::remainder(a, b) <xt::remainder>`                |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.minimum(a, b) <numpy.minimum>`                    | :cpp:func:`xt::minimum(a, b) <xt::minimum>`                    |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.maximum(a, b) <numpy.maximum>`                    | :cpp:func:`xt::maximum(a, b) <xt::maximum>`                    |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.clip(a, min, max) <numpy.clip>`                   | :cpp:func:`xt::clip(a, min, max) <xt::clip>`                   |
   +------------------------------------------------------------+----------------------------------------------------------------+
   |                                                            | :cpp:func:`xt::fma(a, b, c) <xt::fma>`                         |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.interp(x, xp, fp, [,left, right]) <numpy.interp>` | :cpp:func:`xt::interp(x, xp, fp, [,left, right]) <xt::interp>` |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.rad2deg(a) <numpy.rad2deg>`                       | :cpp:func:`xt::rad2deg(a) <xt::rad2deg>`                       |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.degrees(a) <numpy.degrees>`                       | :cpp:func:`xt::degrees(a) <xt::degrees>`                       |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.deg2rad(a) <numpy.deg2rad>`                       | :cpp:func:`xt::deg2rad(a) <xt::deg2rad>`                       |
   +------------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.radians(a) <numpy.radians>`                       | :cpp:func:`xt::radians(a) <xt::radians>`                       |
   +------------------------------------------------------------+----------------------------------------------------------------+

**Exponential functions:**

.. table::
   :widths: 50 50

   +----------------------------------+--------------------------------------+
   |            Python 3 - NumPy      |                C++ 14 - xtensor      |
   +==================================+======================================+
   | :any:`np.exp(a) <numpy.exp>`     | :cpp:func:`xt::exp(a) <xt::exp>`     |
   +----------------------------------+--------------------------------------+
   | :any:`np.expm1(a) <numpy.expm1>` | :cpp:func:`xt::expm1(a) <xt::expm1>` |
   +----------------------------------+--------------------------------------+
   | :any:`np.log(a) <numpy.log>`     | :cpp:func:`xt::log(a) <xt::log>`     |
   +----------------------------------+--------------------------------------+
   | :any:`np.log1p(a) <numpy.log1p>` | :cpp:func:`xt::log1p(a) <xt::log1p>` |
   +----------------------------------+--------------------------------------+

**Power functions:**

.. table::
   :widths: 50 50

   +-------------------------------------+----------------------------------------+
   |            Python 3 - NumPy         |                C++ 14 - xtensor        |
   +=====================================+========================================+
   | :any:`np.power(a, p) <numpy.power>` | :cpp:func:`xt::pow(a, b) <xt::pow>`    |
   +-------------------------------------+----------------------------------------+
   | :any:`np.sqrt(a) <numpy.sqrt>`      | :cpp:func:`xt::sqrt(a) <xt::sqrt>`     |
   +-------------------------------------+----------------------------------------+
   | :any:`np.square(a) <numpy.square>`  | :cpp:func:`xt::square(a) <xt::square>` |
   |                                     | :cpp:func:`xt::cube(a) <xt::cube>`     |
   +-------------------------------------+----------------------------------------+
   | :any:`np.cbrt(a) <numpy.cbrt>`      | :cpp:func:`xt::cbrt(a) <xt::cbrt>`     |
   +-------------------------------------+----------------------------------------+

**Trigonometric functions:**

.. table::
   :widths: 50 50

   +------------------------------+----------------------------------+
   |            Python 3 - NumPy  |                C++ 14 - xtensor  |
   +==============================+==================================+
   | :any:`np.sin(a) <numpy.sin>` | :cpp:func:`xt::sin(a) <xt::sin>` |
   +------------------------------+----------------------------------+
   | :any:`np.cos(a) <numpy.cos>` | :cpp:func:`xt::cos(a) <xt::cos>` |
   +------------------------------+----------------------------------+
   | :any:`np.tan(a) <numpy.tan>` | :cpp:func:`xt::tan(a) <xt::tan>` |
   +------------------------------+----------------------------------+

**Hyperbolic functions:**

.. table::
   :widths: 50 50

   +--------------------------------+------------------------------------+
   |            Python 3 - NumPy    |                C++ 14 - xtensor    |
   +================================+====================================+
   | :any:`np.sinh(a) <numpy.sinh>` | :cpp:func:`xt::sinh(a) <xt::sinh>` |
   +--------------------------------+------------------------------------+
   | :any:`np.cosh(a) <numpy.cosh>` | :cpp:func:`xt::cosh(a) <xt::cosh>` |
   +--------------------------------+------------------------------------+
   | :any:`np.tanh(a) <numpy.tanh>` | :cpp:func:`xt::tanh(a) <xt::tanh>` |
   +--------------------------------+------------------------------------+

**Error and gamma functions:**

.. table::
   :widths: 50 50

   +---------------------------------------------------------+----------------------------------------+
   |            Python 3 - NumPy                             |                C++ 14 - xtensor        |
   +=========================================================+========================================+
   | :any:`scipy.special.erf(a) <scipy.special.erf>`         | :cpp:func:`xt::erf(a) <xt::erf>`       |
   +---------------------------------------------------------+----------------------------------------+
   | :any:`scipy.special.gamma(a) <scipy.special.gamma>`     | :cpp:func:`xt::tgamma(a) <xt::tgamma>` |
   +---------------------------------------------------------+----------------------------------------+
   | :any:`scipy.special.gammaln(a) <scipy.special.gammaln>` | :cpp:func:`xt::lgamma(a) <xt::lgamma>` |
   +---------------------------------------------------------+----------------------------------------+

**Classification functions:**

.. table::
   :widths: 50 50

   +-----------------------------------------------------------+----------------------------------------------------------------+
   |            Python 3 - NumPy                               |                C++ 14 - xtensor                                |
   +===========================================================+================================================================+
   | :any:`np.isnan(a) <numpy.isnan>`                          | :cpp:func:`xt::isnan(a) <xt::isnan>`                           |
   +-----------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.isinf(a) <numpy.isinf>`                          | :cpp:func:`xt::isinf(a) <xt::isinf>`                           |
   +-----------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.isfinite(a) <numpy.isfinite>`                    | :cpp:func:`xt::isfinite(a) <xt::isfinite>`                     |
   +-----------------------------------------------------------+----------------------------------------------------------------+
   | :any:`np.searchsorted(a, v[, side]) <numpy.searchsorted>` | :cpp:func:`xt::searchsorted(a, v[, right]) <xt::searchsorted>` |
   +-----------------------------------------------------------+----------------------------------------------------------------+

**Histogram:**

.. table::
   :widths: 50 50

   +--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
   |                           Python 3 - NumPy                                                                   |                           C++ 14 - xtensor                                                                       |
   +==============================================================================================================+==================================================================================================================+
   | :any:`np.histogram(a, bins[, weights][, density]) <numpy.histogram>`                                         | :cpp:func:`xt::histogram(a, bins[, weights][, density]) <xt::histogram>`                                         |
   +--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
   | :any:`np.histogram_bin_edges(a, bins[, weights][, left, right][, bins][, mode]) <numpy.histogram_bin_edges>` | :cpp:func:`xt::histogram_bin_edges(a, bins[, weights][, left, right][, bins][, mode]) <xt::histogram_bin_edges>` |
   +--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
   | :any:`np.bincount(arr) <numpy.bincount>`                                                                     | :cpp:func:`xt::bincount(arr) <xt::bincount>`                                                                     |
   +--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
   | :any:`np.digitize(data, bin_edges[, right]) <numpy.digitize>`                                                | :cpp:func:`xt::digitize(data, bin_edges[, right][, assume_sorted]) <xt::digitize>`                               |
   +--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+

See :ref:`histogram`.

**Numerical constants:**

.. table::
   :widths: 50 50

   +------------------+----------------------------------------------------------------------------+
   | Python 3 - NumPy | C++ 14 - xtensor                                                           |
   +==================+============================================================================+
   | :any:`numpy.pi`  | :cpp:var:`xt::numeric_constants\<double\>::PI <xt::numeric_constants::PI>` |
   +------------------+----------------------------------------------------------------------------+

Linear algebra
--------------

Many functions found in the :any:`numpy.linalg` module are implemented in `xtensor-blas`_, a separate package offering BLAS and LAPACK bindings,
as well as a convenient interface replicating the ``linalg`` module.

Please note, however, that while we're trying to be as close to NumPy as possible, some features are not
implemented yet. Most prominently that is broadcasting for all functions except for :cpp:func:`xt::linalg::dot`.


**Matrix, vector and tensor products**

.. table::
   :widths: 50 50

   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   |              Python 3 - NumPy                                     |               C++ 14 - xtensor                                                  |
   +===================================================================+=================================================================================+
   | :any:`np.dot(a, b) <numpy.dot>`                                   | :cpp:func:`xt::linalg::dot(a, b) <xt::linalg::dot>`                             |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.vdot(a, b) <numpy.vdot>`                                 | :cpp:func:`xt::linalg::vdot(a, b) <xt::linalg::vdot>`                           |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.outer(a, b) <numpy.outer>`                               | :cpp:func:`xt::linalg::outer(a, b) <xt::linalg::outer>`                         |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.linalg.matrix_power(a, 123) <numpy.linalg.matrix_power>` | :cpp:func:`xt::linalg::matrix_power(a, 123) <xt::linalg::matrix_power>`         |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.kron(a, b) <numpy.kron>`                                 | :cpp:func:`xt::linalg::kron(a, b) <xt::linalg::kron>`                           |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.tensordot(a, b, axes=3) <numpy.tensordot>`               | :cpp:func:`xt::linalg::tensordot(a, b, 3) <xt::linalg::tensordot>`              |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+
   | :any:`np.tensordot(a, b, axes=((0,2),(1,3)) <numpy.tensordot>`    | :cpp:func:`xt::linalg::tensordot(a, b, {0, 2}, {1, 3}) <xt::linalg::tensordot>` |
   +-------------------------------------------------------------------+---------------------------------------------------------------------------------+


**Decompositions**

.. table::
   :widths: 50 50

   +------------------------------------------------------+------------------------------------------------------------+
   |       Python 3 - NumPy                               |       C++ 14 - xtensor                                     |
   +======================================================+============================================================+
   | :any:`np.linalg.cholesky(a) <numpy.linalg.cholesky>` | :cpp:func:`xt::linalg::cholesky(a) <xt::linalg::cholesky>` |
   +------------------------------------------------------+------------------------------------------------------------+
   | :any:`np.linalg.qr(a) <numpy.linalg.qr>`             | :cpp:func:`xt::linalg::qr(a) <xt::linalg::qr>`             |
   +------------------------------------------------------+------------------------------------------------------------+
   | :any:`np.linalg.svd(a) <numpy.linalg.svd>`           | :cpp:func:`xt::linalg::svd(a) <xt::linalg::svd>`           |
   +------------------------------------------------------+------------------------------------------------------------+


**Matrix eigenvalues**

.. table::
   :widths: 50 50

   +------------------------------------------------------+------------------------------------------------------------+
   |       Python 3 - NumPy                               |       C++ 14 - xtensor                                     |
   +======================================================+============================================================+
   | :any:`np.linalg.eig(a) <numpy.linalg.eig>`           | :cpp:func:`xt::linalg::eig(a) <xt::linalg::eig>`           |
   +------------------------------------------------------+------------------------------------------------------------+
   | :any:`np.linalg.eigvals(a) <numpy.linalg.eigvals>`   | :cpp:func:`xt::linalg::eigvals(a) <xt::linalg::eigvals>`   |
   +------------------------------------------------------+------------------------------------------------------------+
   | :any:`np.linalg.eigh(a) <numpy.linalg.eigh>`         | :cpp:func:`xt::linalg::eigh(a) <xt::linalg::eigh>`         |
   +------------------------------------------------------+------------------------------------------------------------+
   | :any:`np.linalg.eigvalsh(a) <numpy.linalg.eigvalsh>` | :cpp:func:`xt::linalg::eigvalsh(a) <xt::linalg::eigvalsh>` |
   +------------------------------------------------------+------------------------------------------------------------+

**Norms and other numbers**

.. table::
   :widths: 50 50

   +------------------------------------------------------------+------------------------------------------------------------------+
   |        Python 3 - NumPy                                    |        C++ 14 - xtensor                                          |
   +============================================================+==================================================================+
   | :any:`np.linalg.norm(a, order=2) <numpy.linalg.norm>`      | :cpp:func:`xt::linalg::norm(a, 2) <xt::linalg::norm>`            |
   +------------------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.linalg.cond(a) <numpy.linalg.cond>`               | :cpp:func:`xt::linalg::cond(a) <xt::linalg::cond>`               |
   +------------------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.linalg.det(a) <numpy.linalg.det>`                 | :cpp:func:`xt::linalg::det(a) <xt::linalg::det>`                 |
   +------------------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.linalg.matrix_rank(a) <numpy.linalg.matrix_rank>` | :cpp:func:`xt::linalg::matrix_rank(a) <xt::linalg::matrix_rank>` |
   +------------------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.linalg.slogdet(a) <numpy.linalg.slogdet>`         | :cpp:func:`xt::linalg::slogdet(a) <xt::linalg::slogdet>`         |
   +------------------------------------------------------------+------------------------------------------------------------------+
   | :any:`np.trace(a) <numpy.trace>`                           | :cpp:func:`xt::linalg::trace(a) <xt::linalg::trace>`             |
   +------------------------------------------------------------+------------------------------------------------------------------+

**Solving equations and inverting matrices**

.. table::
   :widths: 50 50

   +---------------------------------------------------+---------------------------------------------------------+
   |        Python 3 - NumPy                           |        C++ 14 - xtensor                                 |
   +===================================================+=========================================================+
   | :any:`np.linalg.inv(a) <numpy.linalg.inv>`        | :cpp:func:`xt::linalg::inv(a) <xt::linalg::inv>`        |
   +---------------------------------------------------+---------------------------------------------------------+
   | :any:`np.linalg.pinv(a) <numpy.linalg.pinv>`      | :cpp:func:`xt::linalg::pinv(a) <xt::linalg::pinv>`      |
   +---------------------------------------------------+---------------------------------------------------------+
   | :any:`np.linalg.solve(A, b) <numpy.linalg.solve>` | :cpp:func:`xt::linalg::solve(A, b) <xt::linalg::solve>` |
   +---------------------------------------------------+---------------------------------------------------------+
   | :any:`np.linalg.lstsq(A, b) <numpy.linalg.lstsq>` | :cpp:func:`xt::linalg::lstsq(A, b) <xt::linalg::lstsq>` |
   +---------------------------------------------------+---------------------------------------------------------+


.. _`xtensor-blas`: https://github.com/xtensor-stack/xtensor-blas
