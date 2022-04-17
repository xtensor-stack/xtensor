.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Missing values
==============

*xtensor* handles missing values and provides specialized container types for an optimized support of missing values.

Optional expressions
--------------------

Support of missing values in xtensor is primarily provided through the :cpp:type:`xtl::xoptional`
value type and the :cpp:type:`xt::xtensor_optional` and :cpp:type:`xt::xarray_optional` containers.
In the following example, we instantiate a 2-D tensor with a missing value:

.. code:: cpp

    xt::xtensor_optional<double, 2> m
        {{ 1.0 ,       2.0         },
         { 3.0 , missing<double>() }};

This code is semantically equivalent to

.. code:: cpp

    xt::xtensor<xtl::xoptional<double>, 2> m
        {{ 1.0 ,       2.0         },
         { 3.0 , missing<double>() }};

The :cpp:type:`xt::xtensor_optional` container is optimized to handle missing values.
Internally, instead of holding a single container of optional values, it holds an array of ``double``
and a boolean container where each value occupies a single bit instead of ``sizeof(bool)`` bytes.

The :cpp:type:`xt::xtensor_optional::reference` typedef, which is the return type of
:cpp:func:`~xt::xexpression::operator()` is a reference proxy which can be used as an
lvalue for assigning new values in the array.
It happens to be an instance of :cpp:type:`xtl::xoptional\<T, B\> <xtl::optional>` where ``T`` and
``B`` are actually the reference types of the underlying storage for values and boolean flags.

This technique enables performance improvements in mathematical operations over boolean arrays including SIMD optimizations, and
reduces the memory footprint of optional arrays. It should be transparent to the user.

Operating on missing values
---------------------------

Arithmetic operators and mathematical universal functions are overloaded for optional values so that they can be operated upon in the
same way as regular scalars.

.. code:: cpp

    xt::xtensor_optional<double, 2> a
        {{ 1.0 ,       2.0         },
         { 3.0 , missing<double>() }};

    xt::xtensor<double, 1> b
        { 1.0, 2.0 };

    // ``b`` is broadcasted to match the shape of ``a``
    std::cout << a + b << std::endl;

outputs:

.. code::

    {{  2,   4},
     {  4, N/A}}

Optional assemblies
-------------------

The classes :cpp:type:`xt::xoptional_assembly` and :cpp:type:`xt::xoptional_assembly_adaptor` provide
containers and adaptors holding missing values that are optimized for element-wise operations.
Contrary to :cpp:type:`xt::xtensor_optional` and :cpp:type:`xt::xarray_optional`, the optional
assemblies hold two expressions, one holding the values, the other holding the mask for the missing values.
The difference between :cpp:type:`xt::xoptional_assembly` and :cpp:type:`xt::xoptional_assembly_adaptor`
is that the first one is the owner of the two expressions while the last one holds a reference on at least
one of the two expressions.

.. code:: cpp

    xt::xarray<double> v
        {{ 1.0, 2.0 },
         { 3.0, 4.0 }};

    xt::xarray<bool> hv
        {{ true, true  },
         { true, false }};

    xt::xoptional_assembly<xt::xarray<double>, xt::xarray<bool>> assembly(v, hv);
    std::cout << assembly << std::endl;

outputs:

.. code::

    {{ 1,  2 },
     { 3, N/A}}

Handling expressions with missing values
----------------------------------------

Functions :cpp:func:`xt::has_value(E&& e) <xt::has_value>` and :cpp:func:`xt::value(E&& e) <xt::value>`
return expressions corresponding to the underlying value and flag of optional elements.
When ``e`` is an lvalue, :cpp:func:`xt::has_value(E&& e) <xt::has_value>` and
:cpp:func:`xt::value(E&& e) <xt::value>` are lvalues too.

.. code:: cpp

    xt::xtensor_optional<double, 2> a
        {{ 1.0 ,       2.0         },
         { 3.0 , missing<double>() }};


    xt::xtensor<bool, 2> b = xt::has_value(a);

    std::cout << b << std::endl;

outputs:

.. code::

    {{  true,  true},
     {  true, false}}
