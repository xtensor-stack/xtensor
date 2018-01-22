.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Adapting 1-D containers
=======================

`xtensor` can adapt one-dimensional containers in place, and provide them a tensor interface.
Only random access containers can be adapted.

Adapting std::vector
--------------------

The following example shows how to bring an ``std::vector`` into the expression system of
`xtensor`:

.. code::

    #include <cstddef>
    #include <vector>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xadapt.hpp"

    std::vector<double> v = {1., 2., 3., 4., 5., 6. };
    std::vector<std::size_t> shape = { 2, 3 };
    auto a1 = xt::adapt(v, shape);

    xt::xarray<double> a2 = {{ 1., 2., 3.},
                             { 4., 5., 6.}};

    xt::xarray<double> res = a1 + a2;
    // res = {{ 2., 4., 6. }, { 8., 10., 12. }};

``v`` is not copied into ``a1``, so if you change a value in ``a1``, you're actually changing
the corresponding value in ``v``:

.. code::

    a1(0, 0) = 20.;
    // now v is { 20., 2., 3., 4., 5., 6. }

Adapting C-style arrays
-----------------------

`xtensor` provides two ways for adapting C-style array; the first one does not take the
ownership of the array:

.. code::

    #include <cstddef>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xadapt.hpp"

    double compute(double* data, std::size_t size)
    {
        std::vector<std::size_t> shape = { 2, 3 };
        auto a = xt::adapt(data, size, xt::no_ownership(), shape);
        return some_computation(data, size)
    }

    std::size_t size = get_data_size();
    double* data = new double[size];
    compute(data, size);
    // data is still available here
    std::cout << data[0] << std::endl;

However if you replace ``xt::no_ownership`` with ``xt::acquire_ownership``, the adaptor will take
the ownership of the array, meaning it will be deleted when the adaptor is destroyed:

.. code::

    #include <cstddef>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xadapt.hpp"
    
    double compute(double* data, std::size_t size)
    {
        std::vector<std::size_t> shape = { 2, 3 };
        auto a = xt::adapt(data, size, xt::acquire_ownership(), shape);
        return some_computation(data, size)
    }

    std::size_t size = get_data_size();
    double* data = new double[size];
    compute(data, size);
    // data has been deleted 
    
