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

Although ``xtensor`` is a header-only library, we provide standardized means to
install it, with package managers or with cmake.

Besides the xtensor headers, all these methods place the ``cmake`` project
configuration file in the right location so that third-party projects can use
cmake's ``find_package`` to locate xtensor headers.

.. image:: conda.svg

Using the conda package
-----------------------

A package for xtensor is available on the conda package manager.

.. code::

    conda install -c conda-forge xtensor

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

You can also install ``xtensor`` from source with cmake. This requires that you
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
dependencies and installs libraries. ``xtensor`` installation from cmake assumes
this folder contains ``include`` and ``lib`` subfolders.

See the :doc:`build-options` section for more details about cmake options.

Including xtensor in your project
---------------------------------

The different packages of ``xtensor`` are built with cmake, so whatever the
installation mode you choose, you can add ``xtensor`` to your project using cmake:

.. code::

    find_package(xtensor REQUIRED)
    target_include_directories(your_target PUBLIC ${xtensor_INCLUDE_DIRS})
    target_link_libraries(your_target PUBLIC xtensor)

.. _xtl: https://github.com/QuantStack/xtl
