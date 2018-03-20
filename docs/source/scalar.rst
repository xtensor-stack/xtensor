.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Scalars and 0-D expressions
===========================

Assignment
----------

In ``xtensor``, scalars are handled as if they were 0-dimensional expressions. This means that when assigning
a scalar value to an ``xarray``, this last one is **not filled** with that value, but resized to become a 0-D
array containing the scalar value:

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
    double s = 1.2;
    a = s;
    std::cout << a << std::endl;
    // prints 1.2


While this may look weird and counter-intuitive, this actually ensures full consistency of the expression system.
The easiest way to illustrate this is to assume that we have the intuitive scalar assignment (i.e. a broadcasting
assignment) and see how it breaks consistency.

Copy semantic consistency
-------------------------

Assuming that the scalar assignment does not resize the array, we have the following behavior:

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
    double s = 1.2;
    a = 1.2;
    std::cout << a << std::endl;
    // prints {{1.2, 1.2, 1.2}, {1.2, 1.2, 1.2}}

This is not consistent with the behavior of the copy constructor from a scalar:

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<double> a(1.2);
    std::cout << a << std::endl;
    // prints 1.2 (a is a 0-D array)

A way to fix this is to disable copy construction from scalar, and provide a constructor taking a shape and
a scalar: 

.. code::

    #include "xtensor/xarray.hpp"

    xt::xarray<double> a = {{0., 1., 2.}, {3., 4., 5.}};
    a = 1.2;
    xt::xarray<double> b({2, 3}, 1.2);

Although this looks like an acceptable solution, it actually breaks consistency between scalars and 0-dimensional
expressions. This may lead to vicious bugs as explained in the next section.

Scalar and 0-D expressions
--------------------------

Assume that you need a function that computes the mean of the elements of an expression and stores it in another expression.
A possible implementation is:

.. code::

    template <class E1, class E2>
    void eval_mean(const E1& e1, E2& e2)
    {
        e2 = sum(e1) / e1.size();
    }

Then, somewhere in your program:

.. code::

    // somewhere in the code
    xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}},
    xarray<double> b = a;
    // ...
    // later
    eval_mean(a, b);
    // Now b is a 0-D container holding 21.

After that, ``b`` is a 0-dimensional array containing the mean of the elements of ``a``. Indeed, ``sum(a) / e1.size()`` is a
0-D expression, thus when assigned to ``b``, this latter is resized. Later, you realize that you also need the sum of the elements
of ``a``. Since the ``eval_mean`` function already computes it, you decide to return it from that function:

.. code::

    template <class E1, class E2>
    double eval_mean(const E1& e1, E2& e2)
    {
        double s = sum(e1)();
        e2 = s / e1.size();
        return s;
    }

And then you change the client code:

.. code::

    // somewhere in the code
    xarray<double> a = {{1., 2., 3.}, {4., 5., 6.}},
    xarray<double> b = a;
    // ...
    // later
    double s = eval_mean(a, b);
    // Now b is a 2-D container!

After that, ``b`` has become a 2-dimensional array! Indeed, since assigning a scalar to an expression does not resize it, the change in
``eval_mean`` implementation now assigns the mean of ``a`` to each elements of ``b``.

This simple example shows that without consistency between scalars and 0-D expressions, refactoring the code to cache the result
of some 0-D computation actually *silently* changes the shape of the expressions that this result is assigned to.

The only way to avoid that behavior and the bugs it leads to is to handle scalars as if they were 0-dimensional expressions.

