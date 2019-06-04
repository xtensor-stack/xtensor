.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Common pitfalls
===============

xarray initialization
---------------------

.. code::

    xt::xarray<double> a({1, 3, 4, 2});

does not initialize a 4D-array, but a 1D-array containing the values ``1``, ``3``,
``4``, and ``2``. 
It is strictly equivalent to

.. code::

    xt::xarray<double> a = {1, 3, 4, 2};

To initialize a 4D-array with the given shape, use the ``from_shape`` static method:

.. code::

    auto a = xt::xarray<double>::from_shape({1, 3, 4, 2});

The confusion often comes from the way ``xtensor`` can be initialized:

.. code::

    xt::xtensor<double, 4> a = {1, 3, 4, 2};

In this case, a 4D-tensor with shape ``(1, 3, 4, 2)`` is initialized.

Intermediate result
-------------------

Consider the following function:

.. code::

    template <class C>
    auto func(const C& c)
    {
        return (1 - func_tmp(c)) / (1 + func_tmp(c));
    }

where ``func_tmp`` is another unary function accepting an xtensor expression. You may
be tempted to simplify it a bit:

.. code::

    template <class C>
    auto func(const C&)
    {
        auto tmp = func_tmp(c);
        return (1 - tmp) / (1 + tmp);
    }

Unfortunately, you introduced a bug; indeed, expressions in ``xtensor`` are not evaluated
immediately, they capture their arguments by reference or copy depending on their nature,
for future evaluation. Since ``tmp`` is an lvalue, it is captured by reference in the last
statement; when the function returns, ̀`tmp`` is destroyed, leading to a dangling reference
in the returned expression.

Replacing ``auto tmp`` with ``xt::xarray<double> tmp`` does not change anything, ``tmp``
is still an lvalue and thus captured by reference.

