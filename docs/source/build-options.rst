.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Build and configuration
=======================

Build
-----

``xtensor`` build supports the following options:

- ``BUILD_TESTS``: enables the ``xtest`` and ``xbenchmark`` targets (see below).
- ``DOWNLOAD_GTEST``: downloads ``gtest`` and builds it locally instead of using a binary installation.
- ``GTEST_SRC_DIR``: indicates where to find the ``gtest`` sources instead of downloading them.
- ``XTENSOR_ENABLE_ASSERT``: activates the assertions in ``xtensor``.

All these options are disabled by default. Enabling ``DOWNLOAD_GTEST`` or setting ``GTEST_SRC_DIR``
enables ``BUILD_TESTS``.

If the ``BUILD_TESTS`` option is enabled, the following targets are available:

- xtest: builds an run the test suite.
- xbenchmark: builds and runs the benchmarks.

For instance, building the test suite of ``xtensor`` with assertions enabled:

.. code::

    mkdir build
    cd build
    cmake -DBUILD_TESTS=ON -DXTENSOR_ENABLE_ASSERT=ON ../
    make xtest

Building the test suite of ``xtensor`` where the sources of ``gtest`` are located in e.g. ``/usr/share/gtest``:

.. code::

    mkdir build
    cd build
    cmake -DGTEST_SRC_DIR=/usr/share/gtest ../
    make xtest


Configuration
-------------

``xtensor`` can be configured via macros, which must be defined *before* including any of its header. Here is a list of
available macros:

- ``XTENSOR_ENABLE_ASSERT``: enables assertions in xtensor, such as bound check.
- ``DEFAULT_DATA_CONTAINER(T, A)``: defines the type used as the default data container for tensor and arrays. ``T``
  is the ``value_type`` of the container and ``A`` its ``allocator_type``.
- ``DEFAULT_SHAPE_CONTAINER(T, EA, SA)``: defines the type used as the default shape container for tensor and arrays.
  ``T`` is the ``value_type`` of the data container, ``EA`` its ``allocator_type``, and ``SA`` is the ``allocator_type``
  of the shape container.


