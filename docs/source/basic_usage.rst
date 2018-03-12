.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Basic usage
===========

**Initialize a 2-D array and compute the sum of one of its rows and a 1-D array.**

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"

    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2
      {5.0, 6.0, 7.0};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    std::cout << res;

Outputs:

.. code::

   {7, 11, 14}

**Initialize a 1-D array and reshape it in-place.**

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"

    xt::xarray<int> arr
      {1, 2, 3, 4, 5, 6, 7, 8, 9};

    arr.reshape({3, 3});

    std::cout << arr;

Outputs:

.. code::

    {{1, 2, 3},
     {4, 5, 6},
     {7, 8, 9}}

**Index Access**

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"

    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};
    
    std::cout << arr1(0, 0) << std::endl;

    xt::xarray<int> arr2
      {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::cout << arr2(0);

Outputs:

.. code::

    1.0
    1
     
**Broadcasting the** ``xt::pow`` **universal functions.**

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xmath.hpp"
    #include "xtensor/xio.hpp"

    xt::xarray<double> arr1
      {1.0, 2.0, 3.0};

    xt::xarray<unsigned int> arr2
      {4, 5, 6, 7};

    arr2.reshape({4, 1});

    xt::xarray<double> res = xt::pow(arr1, arr2);

    std::cout << res;

Outputs:

.. code::

    {{1, 16, 81},
     {1, 32, 243},
     {1, 64, 729},
     {1, 128, 2187}}

