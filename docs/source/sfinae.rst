.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _sfinae:

SFINAE
======

Rank overload
-------------

All `xtensor`'s classes have a member ``rank`` that can be used
to overload based on rank using *SFINAE*.
Consider the following example:

.. code-block:: cpp

    template <class E, std::enable_if_t<!xt::has_rank_t<E, 2>::value, int> = 0>
    inline E foo(E&& a)
    {
        ... // act on object of flexible rank, or fixed rank != 2
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
    inline E foo(E&& a)
    {
        ... // act on object of fixed rank == 2
    }

    TEST(sfinae, rank_basic)
    {
        xt::xarray<size_t> a = {{9, 9}, {9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};

        foo(a); // flexible rank -> first overload
        foo(b); // fixed rank == 2 -> first overload
        foo(c); // fixed rank == 2 -> second overload
    }

.. note::

    If one wants to test for more than a single value for ``rank``,
    one can use the default value ``SIZE_MAX`` used for flexible rank objects.
    For example, one could have the following overloads:

    .. code-block:: cpp

        // flexible rank
        template <class E, std::enable_if_t<xt::has_rank_t<E, SIZE_MAX>::value, int> = 0>
        inline E foo(E&& a);

        // fixed rank == 1
        template <class E, std::enable_if_t<xt::has_rank_t<E, 1>::value, int> = 0>
        inline E foo(E&& a);

        // fixed rank == 2
        template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
        inline E foo(E&& a);

    Note that fixed ranks other than 1 and 2 will raise a compiler error.

    Of course, if one wants a more limited scope, one could also do the following:

    .. code-block:: cpp

        // flexible rank
        inline void foo(xt::xarray<double>& a);

        // fixed rank == 1
        inline void foo(xt::xtensor<double,1>& a);

        // fixed rank == 2
        inline void foo(xt::xtensor<double,2>& a);
