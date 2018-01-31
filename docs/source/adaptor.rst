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
    #include "xtensor/xadapt.hpp"

    void compute(double* data, std::size_t size)
    {
        std::vector<std::size_t> shape = { size };
        auto a = xt::adapt(data, size, xt::no_ownership(), shape);
        a = a + a; // does not modify the size
    }

    int main()
    {
        std::size_t size = 2;
        double* data = new double[size];
        for (int i = 0; i < size; i++)
            data[i] = i;
        std::cout << data << std::endl;
        // prints e.g. 0x557a363b7c20
        compute(data, size);
        std::cout << data << std::endl;
        // prints e.g. 0x557a363b7c20 (same pointer)
        for (int i = 0; i < size; i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;
        // prints 0 2 (data is still available here)
    }

However if you replace ``xt::no_ownership`` with ``xt::acquire_ownership``, the adaptor will take
the ownership of the array, meaning it will be deleted when the adaptor is destroyed:

.. code::

    #include <cstddef>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xadapt.hpp"

    void compute(double*& data, std::size_t size)
    {
        // data pointer can be changed, hence double*&
        std::vector<std::size_t> shape = { size };
        auto a = xt::adapt(data, size, xt::acquire_ownership(), shape);
        xt::xarray<double> b {1., 2.};
        b.reshape({2, 1});
        a = a * b; // size has changed, shape is now { 2, 2 }
    }

    int main()
    {
        std::size_t size = 2;
        double* data = new double[size];
        for (int i = 0; i < size; i++)
            data[i] = i;
        std::cout << data << std::endl;
        // prints e.g. 0x557a363b7c20
        compute(data, size);
        std::cout << data << std::endl;
        // prints e.g. 0x557a363b8220 (pointer has changed)
        for (int i = 0; i < size * size; i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;
        // prints e.g. 4.65504e-310 1 0 2 (data has been deleted and is now corrupted)
    }

To safely get the computed data out of the function, you could pass an additional output parameter
to ``compute`` in which you copy the result before exiting the function. Or you can create the
adaptor before calling ``compute`` and pass it to the function:

.. code::

    #include <cstddef>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xadapt.hpp"
    
    template <class A>
    void compute(A& a)
    {
        xt::xarray<double> b {1., 2.};
        b.reshape({2, 1});
        a = a * b; // size has changed, shape is now { 2, 2 }
    }
    
    int main()
    {
        std::size_t size = 2;
        double* data = new double[size];
        for (int i = 0; i < size; i++)
            data[i] = i;
        std::vector<std::size_t> shape = { size };
        auto a = xt::adapt(data, size, xt::acquire_ownership(), shape);
        compute(a);
        for (int i = 0; i < size * size; i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;
        // prints 0 1 0 2
    }
