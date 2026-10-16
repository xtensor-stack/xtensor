.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Installation
============

Although *xtensor* is a header-only library, we provide standardized means to
install it, with package managers or with cmake.

Besides the xtensor headers, all these methods place the ``cmake`` project
configuration file in the right location so that third-party projects can use
cmake's ``find_package`` to locate xtensor headers.

.. image:: conda.svg

Using the conda-forge package
-----------------------------

A package for xtensor is available on the mamba (or conda) package manager.

.. code::

    mamba install -c conda-forge xtensor

.. image:: debian.svg

Using the Debian package
------------------------

A package for xtensor is available on Debian.

.. code::

    sudo apt-get install xtensor-dev

.. image:: spack.svg

Using the Spack package
-----------------------

A package for xtensor is available on the Spack package manager.

.. code::

    spack install xtensor
    spack load --dependencies xtensor

.. image:: cmake.svg

From source with cmake
----------------------

You can also install *xtensor* from source with cmake. This requires that you
have the xtl_ library installed on your system. On Unix platforms, from the
source directory:

.. code::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    make install

On Windows platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    nmake
    nmake install

``path_to_prefix`` is the absolute path to the folder where cmake searches for
dependencies and installs libraries. *xtensor* installation from cmake assumes
this folder contains ``include`` and ``lib`` subfolders.

See the :doc:`build-options` section for more details about cmake options.

Although not officially supported, *xtensor* can be installed with MinGW:

.. code::

    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    mingw32-make
    mingw32-make install

Including xtensor in your project
---------------------------------

The different packages of *xtensor* are built with cmake, so whatever the
installation mode you choose, you can add *xtensor* to your project using cmake:

.. code::

    find_package(xtensor REQUIRED)
    target_include_directories(your_target PUBLIC ${xtensor_INCLUDE_DIRS})
    target_link_libraries(your_target PUBLIC xtensor)

.. _header-migration:

Migrating from xtensor 0.25 or earlier
--------------------------------------

Starting with xtensor 0.26, headers are organized into subdirectories by
functionality. Includes such as ``#include <xtensor/xarray.hpp>`` must be
updated to ``#include <xtensor/containers/xarray.hpp>``. The same change applies
to includes written with double quotes.

Common replacements are:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Before 0.26
     - Since 0.26
   * - ``xtensor/xarray.hpp``
     - ``xtensor/containers/xarray.hpp``
   * - ``xtensor/xtensor.hpp``
     - ``xtensor/containers/xtensor.hpp``
   * - ``xtensor/xfixed.hpp``
     - ``xtensor/containers/xfixed.hpp``
   * - ``xtensor/xadapt.hpp``
     - ``xtensor/containers/xadapt.hpp``
   * - ``xtensor/xmath.hpp``
     - ``xtensor/core/xmath.hpp``
   * - ``xtensor/xeval.hpp``
     - ``xtensor/core/xeval.hpp``
   * - ``xtensor/xview.hpp``
     - ``xtensor/views/xview.hpp``
   * - ``xtensor/xstrided_view.hpp``
     - ``xtensor/views/xstrided_view.hpp``
   * - ``xtensor/xbuilder.hpp``
     - ``xtensor/generators/xbuilder.hpp``
   * - ``xtensor/xrandom.hpp``
     - ``xtensor/generators/xrandom.hpp``
   * - ``xtensor/xio.hpp``
     - ``xtensor/io/xio.hpp``
   * - ``xtensor/xnpy.hpp``
     - ``xtensor/io/xnpy.hpp``
   * - ``xtensor/xmanipulation.hpp``
     - ``xtensor/misc/xmanipulation.hpp``
   * - ``xtensor/xsort.hpp``
     - ``xtensor/misc/xsort.hpp``
   * - ``xtensor/xreducer.hpp``
     - ``xtensor/reducers/xreducer.hpp``
   * - ``xtensor/xnorm.hpp``
     - ``xtensor/reducers/xnorm.hpp``

For other headers, locate the filename under the installed ``include/xtensor``
directory and use its path relative to ``include``. Update includes in your
own headers as well as source files. Keep the compiler's include search path
pointing to the parent of ``xtensor`` (typically ``<prefix>/include``); adding
each new subdirectory to the search path does not replace updating the
``#include`` directives. The CMake integration above supplies the include path.

Also check the language standard and xtl dependency when upgrading:

.. list-table::
   :header-rows: 1

   * - xtensor release
     - C++ standard
     - Minimum xtl version
   * - 0.25.0
     - C++14
     - 0.7.5
   * - 0.26.0
     - C++17
     - 0.8.0
   * - 0.27.0
     - C++20
     - 0.8.0

This addresses the header layout change; consult the :doc:`changelog` for other
API and dependency changes in the release you are upgrading to.

.. _xtl: https://github.com/xtensor-stack/xtl
