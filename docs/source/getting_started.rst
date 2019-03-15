.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Getting started
===============

This short guide explains how to get started with `xtensor` once you have installed it with one of
the methods described in the installation section.

First example
-------------

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"
    #include "xtensor/xview.hpp"

    int main(int argc, char* argv[])
    {
        xt::xarray<double> arr1
          {{1.0, 2.0, 3.0},
           {2.0, 5.0, 7.0},
           {2.0, 5.0, 7.0}};

        xt::xarray<double> arr2
          {5.0, 6.0, 7.0};

        xt::xarray<double> res = xt::view(arr1, 1) + arr2;

        std::cout << res;

        return 0;
    }

This example simply adds the second row of a 2-dimensional array with a 1-dimensional
array.

Compiling the first example
---------------------------

`xtensor` is a header-only library, so there is no library to link with. The only constraint
is that the compiler must be able to find the headers of `xtensor`, this is usually done
by having the directory containing the headers in the include path. With GCC, use the ``-I`` option
to achieve this. Assuming the first example code is located in ``example.cpp``, the compilation command
is:

.. code:: bash

    gcc -I /path/to/xtensor/ example.cpp -o example

When you run the program, it produces the following output:

.. code::

   {7, 11, 14}

Building with cmake
-------------------

A better alternative for building programs using `xtensor` is to use `cmake`, especially if you are
developing for several platforms. Assuming the following folder structure:

.. code:: bash

    first_example
       |- src
       |   |- example.cpp
       |- CMakeLists.txt

The following minimal ``CMakeLists.txt`` is enough to build the first example:

.. code:: cmake

    cmake_minimum_required(VERSION 3.1)
    project(first_example)

    find_package(xtl REQUIRED)
    find_package(xtensor REQUIRED)

    add_executable(first_example src/example.cpp)
    
    if(MSVC)
        target_compile_options(first_example PRIVATE /EHsc /MP /bigobj)
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
        CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
        (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
        target_compile_options(first_example PRIVATE -march=native -std=c++14)
    endif()

    target_link_libraries(first_example xtensor)

`cmake` has to know where to find the headers, this is done through the ``CMAKE_INSTALL_PREFIX``
variable. Note that ``CMAKE_INSTALL_PREFIX`` is usually the path to a folder containing the following
subfolders: ``include``, ``lib`` and ``bin``, so you don't have to pass any additional option for linking.
Examples of valid values for ``CMAKE_INSTALL_PREFIX`` on Unix platforms are ``/usr/local``, ``/opt``.

The following commands create a directory for building (avoid building in the source folder), builds
the first example with cmake and then runs the program:

.. code:: bash

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=your_prefix ..
    make
    ./first_program

See :ref:`build-configuration` for more details about the build options.

Second example: reshape
-----------------------

This second example initializes a 1-dimensional array and reshapes it in-place:

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"

    int main(int argc, char* argv[])
    {
        xt::xarray<int> arr
          {1, 2, 3, 4, 5, 6, 7, 8, 9};

        arr.reshape({3, 3});

        std::cout << arr;
        return 0;
    }

When compiled and run, this produces the following output:

.. code::

    {{1, 2, 3},
     {4, 5, 6},
     {7, 8, 9}}

Third example: index access
---------------------------

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xio.hpp"

    int main(int argc, char* argv[])
    {
        xt::xarray<double> arr1
          {{1.0, 2.0, 3.0},
           {2.0, 5.0, 7.0},
           {2.0, 5.0, 7.0}};

        std::cout << arr1(0, 0) << std::endl;

        xt::xarray<int> arr2
          {1, 2, 3, 4, 5, 6, 7, 8, 9};

        std::cout << arr2(0);
        return 0;
    }

Outputs:

.. code::

    1.0
    1

Fourth example: broadcasting
----------------------------

This last example shows how to broadcast the ``xt::pow`` universal function:

.. code::

    #include <iostream>
    #include "xtensor/xarray.hpp"
    #include "xtensor/xmath.hpp"
    #include "xtensor/xio.hpp"

    int main(int argc, char* argv[])
    {
        xt::xarray<double> arr1
          {1.0, 2.0, 3.0};

        xt::xarray<unsigned int> arr2
          {4, 5, 6, 7};

        arr2.reshape({4, 1});

        xt::xarray<double> res = xt::pow(arr1, arr2);

        std::cout << res;
        return 0;
    }

Outputs:

.. code::

    {{1, 16, 81},
     {1, 32, 243},
     {1, 64, 729},
     {1, 128, 2187}}

