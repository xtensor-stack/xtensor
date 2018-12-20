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

Array indices
-------------

Functions like ``xt::argwhere(a < 5)`` return a ``std::vector`` of *array indices*. Using the same matrix as above, we can do

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

Here we observe that to work print we need to convert the ``std::vector`` to a ``xt::xtensor<size_t, 2>`` array, which is done using ``xt::from_indices``.

From array indices to flat indices
----------------------------------

To convert the array indices to a ``xt::xtensor<size_t, 1>`` of flat indices, ``xt::ravel_indices`` can be used. For to same example:

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

For 1-D arrays the array indices and flat indices coincide. One can use the generic functions ``xt::flatten_indices`` to get a ``xt::xtensor<size_t, 1>`` of (array/flat) indices. For example:

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

which print the indices and the selection (which are in this case identical):

.. code-block:: none

    { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15}
    { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15}

From flat indices to array indices
----------------------------------

To convert *flat indices* to *array_indices* the function ``xt::ravel_indices`` can be used. For example

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

Notice that once again the function ``xt::from_indices`` has been used to convert a ``std::vector`` of indices to a ``xt::xtensor`` array for printing.
