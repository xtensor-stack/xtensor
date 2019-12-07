.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _build-configuration:

Build and configuration
=======================

Configuration
-------------

``xtensor`` can be configured via macros which must be defined *before* including
any of its headers. This can be achieved the following ways:

- either define them in the CMakeLists of your project, with ``target_compile_definitions``
  cmake command.
- or create a header where you define all the macros you want and then include the headers you
  need. Then include this header whenever you need ``xtensor`` in your project.

The following macros are already defined in ``xtensor`` but can be overwritten:

- ``XTENSOR_DEFAULT_DATA_CONTAINER(T, A)``: defines the type used as the default data container for tensors and arrays. ``T``
  is the ``value_type`` of the container and ``A`` its ``allocator_type``.
- ``XTENSOR_DEFAULT_SHAPE_CONTAINER(T, EA, SA)``: defines the type used as the default shape container for tensors and arrays.
  ``T`` is the ``value_type`` of the data container, ``EA`` its ``allocator_type``, and ``SA`` is the ``allocator_type``
  of the shape container.
- ``XTENSOR_DEFAULT_LAYOUT``: defines the default layout (row_major, column_major, dynamic) for tensors and arrays. We *strongly*
  discourage using this macro, which is provided for testing purpose. Prefer defining alias types on tensor and array
  containers instead.
- ``XTENSOR_DEFAULT_TRAVERSAL``: defines the default traversal order (row_major, column_major) for algorithms and iterators on tensors
  and arrays. We *strongly* discourage using this macro, which is provided for testing purpose.

The following macros are helpers for debugging, they are not defined by default:

- ``XTENSOR_ENABLE_ASSERT``: enables assertions in xtensor, such as bound check.
- ``XTENSOR_ENABLE_CHECK_DIMENSION``: enables the dimensions check in ``xtensor``. Note that this option should not be turned
  on if you expect ``operator()`` to perform broadcasting.

External dependencies
---------------------

The last group of macros is for using external libraries to achieve maximum performance (see next section for additional
requirements):

- ``XTENSOR_USE_XSIMD``: enables SIMD acceleration in ``xtensor``. This requires that you have xsimd_ installed
  on your system.
- ``XTENSOR_USE_TBB``: enables parallel assignment loop. This requires that you have tbb_ installed
  on your system.
- ``XTENSOR_DISABLE_EXCEPTIONS``: disables c++ exceptions.
- ``XTENSOR_USE_OPENMP``: enables parallel assignment loop using OpenMP. This requires that OpenMP is available on your system.

Defining these macros in the CMakeLists of your project before searching for ``xtensor`` will trigger automatic finding
of dependencies, so you don't have to include the ``find_package(xsimd)`` and ``find_package(TBB)`` commands in your
CMakeLists:

.. code:: cmake

    set(XTENSOR_USE_XSIMD 1)
    set(XTENSOR_USE_TBB 1)
    # xsimd and TBB dependencies are automatically
    # searched when the following is executed
    find_package(xtensor REQUIRED)

    # the target now sets the proper defines (e.g. "XTENSOR_USE_XSIMD")
    target_link_libraries(... xtensor)


Build and optimization
----------------------

Windows
~~~~~~~

Windows users must activate the ``/bigobj`` flag, otherwise it's almost certain that the compilation fails. More generally,
the following options are recommended:

.. code:: cmake

    target_link_libraries(... xtensor xtensor::optimize)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)

    # OR

    target_compile_options(target_name PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)

If you defined ``XTENSOR_USE_XSIMD``, you must also specify which instruction set you target:

.. code:: cmake

    target_compile_options(target_name PRIVATE /arch:AVX2)
    # OR
    target_compile_options(target_name PRIVATE /arch:AVX)
    # OR
    target_compile_options(target_name PRIVATE /arch:ARMv7VE)

If you build on an old system that does not support any of these instruction sets, you don't have to specify
anything, the system will do its best to enable the most recent supported instruction set.

Linux/OSX
~~~~~~~~~

Whether you enabled ``XTENSOR_USE_XSIMD`` or not, it is highly recommended to build with ``-march=native`` option:

.. code:: cmake

    target_link_libraries(... xtensor xtensor::optimize)

    # OR

    target_compile_options(target_name PRIVATE -march=native)

Notice that this option prevents building on a machine and distributing the resulting binary on another machine with
a different architecture (i.e. not supporting the same instruction set).

.. _xsimd: https://github.com/xtensor-stack/xsimd
.. _tbb: https://www.threadingbuildingblocks.org
