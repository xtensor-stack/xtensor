.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Handling of missing values
==========================

``xtensor`` supports missing values and comprises specialized container types for an optimized support of missing values.

Missing values
--------------

Support of missing values in xtensor is primarily provided through the ``xoptional`` value type and the ``xtensor_optional`` and
``xarray_optional`` containers. In the following example, we instantiate a 2-D tensor with a missing value:

.. code:: cpp

    xtensor_optional<double, 2> m
        {{ 1.0 ,       2.0          },
         { 3.0 , missing<double>()} };

This code is semantically equivalent to

.. code:: cpp

    xtensor<xoptional<double>, 2> m
        {{ 1.0 ,       2.0          },
         { 3.0 , missing<double>()} };

The ``xtensor_optional`` container is optimized to handle missing values. Internally, instead of holding a single container
of optional values, it holds an array of ``double`` and a boolean container where each value occupies a single bit instead of ``sizeof(bool)``
bytes.

The ``xtensor_optional::reference`` typedef, which is the return type of ``operator()`` is a reference proxy which can be used as an
lvalue for assigning new values in the array. It happens to be an instance of ``xoptional<T, B>`` where ``T`` and ``B`` are actually
the reference types of the underlying storage for values and boolean flags.

This technique enables performance improvements in mathematical operations over boolean arrays including SIMD optimizations, and
reduces the memory footprint of optional arrays. It should be transparent to the user.
