.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Operators
=========

Operations and functions of ``xtensor`` are not evaluated until they are assigned.
In the following, ``e1``, ``e2`` and ``e3`` can be arbitrary tensor expressions.

Arithmetic operators
--------------------

.. code::

    auto res0 = -e1;
    auto res1 = e1 + e2;
    auto res2 = e1 - e2;
    auto res3 = e1 * e2;
    auto res4 = e1 / e2;
    auto res5 = e1 % e2;

    res1 += e2;
    res2 -= e2;
    res3 *= e2;
    res4 /= e2;
    res5 %= e2;

Bitwise operators
-----------------

.. code::

    auto res0 = e1 & e2;
    auto res1 = e1 | e2;
    auto res2 = e1 ^ e2;
    auto res3 = ~e1;

    res0 &= e2;
    res1 |= e2;

Logical operators
-----------------

.. code::

    auto res0 = e1 && e2;
    auto res1 = e1 || e2;
    auto res2 = !e1;
    bool res3 = any(e1);
    bool res4 = all(e1);
    auto res5 = where(e1, e2, e3);

Comparison operators
--------------------

Comparison operators return expressions performing element-wise
comparison:

.. code::

    auto res0 = e1 < e2;
    auto res1 = e1 > e2;
    auto res2 = e1 <= e2;
    auto res3 = e1 >= e2;
    auto res4 = xt::equal(e1, e2);
    auto res5 = xt::not_equal(e1, e2);

Except for equality and inequality operators which performs traditional
comparison and return a boolean:

.. code::

    bool res0 = e1 == e2; // true if all elements in e1 equal those in e2
    bool res1 = e1 != e2;

