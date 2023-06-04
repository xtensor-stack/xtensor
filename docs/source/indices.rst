.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Indices
=======

Definition
----------

There are two types of indices: *array indices* and *flat indices*. Consider this example (stored in row-major):

.. code-block:: cpp

    #include <xtensor/xtensor.hpp>
    #include <xtensor/xarray.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xarray<size_t> a = xt::arange<size_t>(3 * 4);

        a.reshape({3,4});

        std::cout << a << std::endl;
    }

Which prints

.. code-block:: none

    {{ 0,  1,  2,  3},
     { 4,  5,  6,  7},
     { 8,  9, 10, 11}}

The *array index* ``{1, 2}`` corresponds to the *flat index* ``6``.

Operators: array index
------------------------

An *array index* can be specified to an operators by a sequence of numbers.
To this end the following operators are at your disposal:

:cpp:func:`operator()(args...) <xt::xcontainer::operator>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a(1, 2) == 6``.
*   See also: :cpp:func:`xt::xcontainer::operator()`.

Returns a (constant) reference to the element,
specified by an *array index* given by a number of unsigned integers.

*   If the number of indices is less that the dimension of the array,
    the indices are pre-padded with zeros until the dimension is matched
    (example: ``a(2) == a(0, 2) == 2``).

*   If the number of indices is greater than the dimension of the array,
    the first ``#indices - dimension`` indices are ignored.

*   To post-pad an arbitrary number of zeros use ``xt::missing``
    (example ``a(2, xt::missing) == a(2, 0) == 8``.

:cpp:func:`at(args...) <xt::xcontainer::at>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a.at(1, 2) == 6``.
*   See also: :cpp:func:`xt::xcontainer::at`.

Same as :cpp:func:`~xt::xcontainer::operator()`:
Returns a (constant) reference to the element,
specified by an *array index* given by a number of unsigned integers.

:cpp:func:`unchecked(args...) <xt::xcontainer::unchecked>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a.unchecked(1, 2) == 6``.
*   See also: :cpp:func:`xt::xcontainer::unchecked`.

Returns a (constant) reference to the element,
specified by an *array index* given by a number of unsigned integers.
Different than :cpp:func:`~xt::xcontainer::operator()` there are no bounds checks (even when assertions)
are turned on, and the number of indices is assumed to match the dimension of the array.
:cpp:func:`~xt::xcontainer::unchecked` is thus aimed at performance.

.. note::

    If you assume responsibility for bounds-checking, this operator can be used to virtually
    post-pad zeros if you specify less indices than the rank of the array.
    Example: ``a.unchecked(1) == a(1, 0)``.

:cpp:func:`periodic(args...) <xt::xcontainer::periodic>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a.periodic(-1, -2) == 7``.
*   See also: :cpp:func:`xt::xcontainer::periodic`.

Returns a (constant) reference to the element,
specified by an *array index* given by a number of signed integers.
Negative and 'overflowing' indices are changed by assuming periodicity along that axis.
For example, for the first axis: ``-1 -> a.shape(0) - 1 = 2``,
likewise for example ``3 -> 3 - a.shape(0) = 0``.
Of course this comes as the cost of some extra complexity.

:cpp:func:`in_bounds(args...) <xt::xcontainer::in_bounds>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a.in_bounds(1, 2) == true``.
*   See also: :cpp:func:`xt::xcontainer::in_bounds`.

Check if the *array index* is 'in bounds', return ``false`` otherwise.

:cpp:func:`operator[]({...}) <xt::xcontainer::operator[]>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a[{1, 2}] == 6``.
*   See also: :cpp:func:`xt::xcontainer::operator[]`.

Returns a (constant) reference to the element,
specified by an *array index* given by a list of unsigned integers.

Operators: flat index
---------------------

:cpp:func:`flat(i) <xt::xcontainer::flat>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   Example: ``a.flat(6) == 6``.
*   See also: :cpp:func:`xt::xcontainer::flat`.

Returns a (constant) reference to the element specified by a *flat index*,
given an unsigned integer.

.. note::

    If the layout would not have been the default *row major*,
    but *column major*, then ``a.flat(6) == 2``.

.. note::

    In many cases ``a.flat(i) == a.data()[i]``.

Array indices
-------------

Functions like :cpp:func:`xt::argwhere(a \< 5) <xt::argwhere>` return a ``std::vector`` of *array indices*.
Using the same matrix as above, we can do

.. code-block:: cpp

    int main()
    {
        xt::xarray<size_t> a = xt::arange<size_t>(3 * 4);

        a.reshape({3,4});

        auto idx = xt::from_indices(xt::argwhere(a >= 6));

        std::cout << idx << std::endl;
    }

which prints

.. code-block:: none

    {{1, 2},
     {1, 3},
     {2, 0},
     {2, 1},
     {2, 2},
     {2, 3}}

To print the ``std::vector``, it is converted to a :cpp:type:`xt::xtensor\<size_t, 2\> <xt::xtensor>`
array, which is done using :cpp:func:`xt::from_indices`.

From array indices to flat indices
----------------------------------

To convert the array indices to a :cpp:type:`xt::xtensor\<size_t, 1\> <xt::xtensor>` of flat indices,
:cpp:func:`xt::ravel_indices` can be used.
For the same example:

.. code-block:: cpp

    #include <xtensor/xtensor.hpp>
    #include <xtensor/xarray.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xarray<size_t> a = xt::arange<size_t>(3 * 4);

        a.reshape({3,4});

        auto idx = xt::ravel_indices(xt::argwhere(a >= 6), a.shape());

        std::cout << idx << std::endl;
    }

which prints

.. code-block:: none

    { 6,  7,  8,  9, 10, 11}

.. note::

    To convert to a ``std::vector`` use

    .. code-block:: cpp

        auto idx = xt::ravel_indices<xt::ravel_vector_tag>(xt::argwhere(a >= 6), a.shape());

1-D arrays: array indices == flat indices
-----------------------------------------

For 1-D arrays the array indices and flat indices coincide.
One can use the generic functions :cpp:func:`xt::flatten_indices` to get a
:cpp:type:`xt::xtensor\<size_t, 1\> <xt::xtensor>` of (array/flat) indices.
For example:

.. code-block:: cpp

    #include <xtensor/xtensor.hpp>
    #include <xtensor/xview.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xtensor<size_t, 1> a = xt::arange<size_t>(16);

        auto idx = xt::flatten_indices(xt::argwhere(a >= 6));

        std::cout << idx << std::endl;

        std::cout << xt::view(a, xt::keep(idx)) << std::endl;
    }

which prints the indices and the selection (which are in this case identical):

.. code-block:: none

    { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15}
    { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15}

From flat indices to array indices
----------------------------------

To convert *flat indices* to *array_indices* the function :cpp:func:`xt::unravel_indices` can be used.
For example

.. code-block:: cpp

    #include <xtensor/xarray.hpp>
    #include <xtensor/xtensor.hpp>
    #include <xtensor/xstrides.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xarray<size_t> a = xt::arange<size_t>(3 * 4);

        a.reshape({3,4});

        auto flat_indices = xt::ravel_indices(xt::argwhere(a >= 6), a.shape());

        auto array_indices = xt::from_indices(xt::unravel_indices(flat_indices, a.shape()));

        std::cout << "flat_indices = " << std::endl << flat_indices << std::endl;
        std::cout << "array_indices = " << std::endl << array_indices << std::endl;
    }

which prints

.. code-block:: none

    flat_indices =
    { 6,  7,  8,  9, 10, 11}
    array_indices =
    {{1, 2},
     {1, 3},
     {2, 0},
     {2, 1},
     {2, 2},
     {2, 3}}

Notice that once again the function :cpp:func:`xt::from_indices` has been used to convert a
``std::vector`` of indices to a :cpp:type:`xt::xtensor` array for printing.
