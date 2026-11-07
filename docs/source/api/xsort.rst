.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xsort
=====

Defined in ``xtensor/misc/xsort.hpp``

Medians over multiple axes
--------------------------

Pass a sequence of axes to ``xt::median`` to group their elements before taking
one median per output element. The remaining axes retain their original order.
For example::

    xt::xarray<double> a = {{{0., 0., 0., 100.}},
                           {{10., 10., 10., 10.}}};
    auto result = xt::median(a, {0, 2});  // shape {1}, value {10.}

Taking successive medians instead gives ``{5.}``. Materialize the intermediate
result before calling the single-axis overload again::

    xt::xarray<double> intermediate = xt::median(a, 2);
    auto successive = xt::median(intermediate, 0);  // shape {1}, value {5.}

The grouped result is independent of the order of the selected axes. Negative
axes count from the end; repeated or invalid axes are rejected.
An empty sequence keeps the input shape, while selecting every axis returns a
scalar expression. Selecting an axis of length zero is an error.

The input is not modified. This overload follows the result-type rules of the
single-axis overload, including its promotion of integral input. This promotion
also applies to an empty axis sequence and can round large integers. Calling
``xt::median(a)`` without axes retains the existing scalar API and result type.

.. cpp:namespace-push:: xt

.. doxygengroup:: xt_xsort

.. cpp:namespace-pop::
