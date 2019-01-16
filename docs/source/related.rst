.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. raw:: html

   <style>
   h2 {
        display: none;
   }
   </style>

.. _related-projects:

Related projects
================

xtensor-python
--------------

.. image:: xtensor-python.svg
   :alt: xtensor-python

The xtensor-python_ project provides the implementation of container types
compatible with ``xtensor``'s expression system, ``pyarray`` and ``pytensor``
which effectively wrap numpy arrays, allowing operating on numpy arrays
in-place.

Example 1: Use an algorithm of the C++ library on a numpy array in-place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++ code**

.. code::

    #include <numeric>                        // Standard library import for std::accumulate
    #include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
    #include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
    #define FORCE_IMPORT_ARRAY                // numpy C api loading
    #include "xtensor-python/pyarray.hpp"     // Numpy bindings

    double sum_of_sines(xt::pyarray<double> &m)
    {
        auto sines = xt::sin(m);
        // sines does not actually hold any value
        return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
    }

    PYBIND11_PLUGIN(xtensor_python_test)
    {
        xt::import_numpy();
        pybind11::module m("xtensor_python_test", "Test module for xtensor python bindings");

        m.def("sum_of_sines", sum_of_sines,
            "Sum the sines of the input values");

        return m.ptr();
    }

**Python code**

.. code::

    Python Code

    import numpy as np
    import xtensor_python_test as xt

    a = np.arange(15).reshape(3, 5)
    s = xt.sum_of_sines(v)
    s

**Outputs**

.. code::

    1.2853996391883833


Example 2: Create a universal function from a C++ scalar function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++ code**

.. code::

    #include "pybind11/pybind11.h"
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pyvectorize.hpp"
    #include <numeric>
    #include <cmath>

    namespace py = pybind11;

    double scalar_func(double i, double j)
    {
        return std::sin(i) - std::cos(j);
    }

    PYBIND11_PLUGIN(xtensor_python_test)
    {
        xt::import_numpy();
        py::module m("xtensor_python_test", "Test module for xtensor python bindings");

        m.def("vectorized_func", xt::pyvectorize(scalar_func), "");

        return m.ptr();
    }

**Python code**

.. code::

    import numpy as np
    import xtensor_python_test as xt

    x = np.arange(15).reshape(3, 5)
    y = [1, 2, 3, 4, 5]
    z = xt.vectorized_func(x, y)
    z

**Outputs**

.. code::

    [[-0.540302,  1.257618,  1.89929 ,  0.794764, -1.040465],
     [-1.499227,  0.136731,  1.646979,  1.643002,  0.128456],
     [-1.084323, -0.583843,  0.45342 ,  1.073811,  0.706945]]

xtensor-python-cookiecutter
---------------------------

.. image:: xtensor-cookiecutter.svg
   :alt: xtensor-python-cookiecutter
   :width: 50%

The xtensor-python-cookiecutter_ project helps extension authors create Python
extension modules making use of `xtensor`.

It takes care of the initial work of generating a project skeleton with

- A complete setup.py compiling the extension module

A few examples included in the resulting project including

- A universal function defined from C++
- A function making use of an algorithm from the STL on a numpy array
- Unit tests
- The generation of the HTML documentation with sphinx

xtensor-julia
-------------

.. image:: xtensor-julia.svg
   :alt: xtensor-julia

The xtensor-julia_ project provides the implementation of container types
compatible with ``xtensor``'s expression system, ``jlarray`` and ``jltensor``
which effectively wrap Julia arrays, allowing operating on Julia arrays
in-place.

Example 1: Use an algorithm of the C++ library with a Julia array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++ code**

.. code::

    #include <numeric>                        // Standard library import for std::accumulate
    #include <cxx_wrap.hpp>                   // CxxWrap import to define Julia bindings
    #include "xtensor-julia/jltensor.hpp"     // Import the jltensor container definition
    #include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions

    double sum_of_sines(xt::jltensor<double, 2> m)
    {
        auto sines = xt::sin(m);  // sines does not actually hold values.
        return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
    }

    JULIA_CPP_MODULE_BEGIN(registry)
        cxx_wrap::Module mod = registry.create_module("xtensor_julia_test");
        mod.method("sum_of_sines", sum_of_sines);
    JULIA_CPP_MODULE_END

**Julia code**

.. code::

    using xtensor_julia_test

    arr = [[1.0 2.0]
           [3.0 4.0]]

    s = sum_of_sines(arr)
    s

**Outputs**

.. code::

   1.2853996391883833

Example 2: Create a numpy-style universal function from a C++ scalar function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++ code**

.. code::

    #include <cxx_wrap.hpp>
    #include "xtensor-julia/jlvectorize.hpp"

    double scalar_func(double i, double j)
    {
        return std::sin(i) - std::cos(j);
    }

    JULIA_CPP_MODULE_BEGIN(registry)
        cxx_wrap::Module mod = registry.create_module("xtensor_julia_test");
        mod.method("vectorized_func", xt::jlvectorize(scalar_func));
    JULIA_CPP_MODULE_END

**Julia code**

.. code::

    using xtensor_julia_test

    x = [[ 0.0  1.0  2.0  3.0  4.0]
         [ 5.0  6.0  7.0  8.0  9.0]
         [10.0 11.0 12.0 13.0 14.0]]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    z = xt.vectorized_func(x, y)
    z

**Outputs**

.. code::

    [[-0.540302  1.257618  1.89929   0.794764 -1.040465],
     [-1.499227  0.136731  1.646979  1.643002  0.128456],
     [-1.084323 -0.583843  0.45342   1.073811  0.706945]]

xtensor-julia-cookiecutter
--------------------------

.. image:: xtensor-cookiecutter.svg
   :alt: xtensor-julia-cookiecutter
   :width: 50%

The xtensor-julia-cookiecutter_ project helps extension authors create Julia
extension modules making use of `xtensor`.

It takes care of the initial work of generating a project skeleton with

- A complete read-to-use Julia package

A few examples included in the resulting project including

- A numpy-style universal function defined from C++
- A function making use of an algorithm from the STL on a numpy array
- Unit tests
- The generation of the HTML documentation with sphinx

xtensor-r
---------

.. image:: xtensor-r.svg
   :alt: xtensor-r

The xtensor-r_ project provides the implementation of container types
compatible with ``xtensor``'s expression system, ``rarray`` and ``rtensor``
which effectively wrap R arrays, allowing operating on R arrays in-place.

Example 1: Use an algorithm of the C++ library on a R array in-place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++ code**

.. code::

    #include <numeric>                    // Standard library import for std::accumulate
    #include "xtensor/xmath.hpp"          // xtensor import for the C++ universal functions
    #include "xtensor-r/rarray.hpp"       // R bindings
    #include <Rcpp.h>

    using namespace Rcpp;

    // [[Rcpp::plugins(cpp14)]]

    // [[Rcpp::export]]
    double sum_of_sines(xt::rarray<double>& m)
    {
        auto sines = xt::sin(m);  // sines does not actually hold values.
        return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
    }

**R code**

.. code::

    v <- matrix(0:14, nrow=3, ncol=5)
    s <- sum_of_sines(v)
    s

**Outputs**

.. code::

    1.2853996391883833

xtensor-blas
------------

.. image:: xtensor-blas.svg
   :alt: xtensor-blas

The xtensor-blas_ project is an extension to the xtensor library, offering
bindings to BLAS and LAPACK libraries through cxxblas and cxxlapack from the
FLENS project. ``xtensor-blas`` powers the ``xt::linalg`` functionalities,
which are the counterpart to numpy's ``linalg`` module.

xtensor-fftw
------------

.. image:: xtensor-fftw.svg
   :alt: xtensor-fftw

The xtensor-fftw_ project is an extension to the xtensor library, offering
bindings to the fftw library.  ``xtensor-fftw`` powers the ``xt::fftw``
functionalities, which are the counterpart to numpy's ``fft`` module.

Example 1: Calculate a derivative in Fourier space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate the derivative of a (discretized) field in Fourier space, e.g. a sine shaped field ``sin``:

**C++ code**

.. code::

    #include <xtensor-fftw/basic.hpp>   // rfft, irfft
    #include <xtensor-fftw/helper.hpp>  // rfftscale
    #include <xtensor/xarray.hpp>
    #include <xtensor/xbuilder.hpp>     // xt::arange
    #include <xtensor/xmath.hpp>        // xt::sin, cos
    #include <complex>
    #include <xtensor/xio.hpp>

    // generate a sinusoid field
    double dx = M_PI / 100;
    xt::xarray<double> x = xt::arange(0., 2 * M_PI, dx);
    xt::xarray<double> sin = xt::sin(x);

    // transform to Fourier space
    auto sin_fs = xt::fftw::rfft(sin);

    // multiply by i*k
    std::complex<double> i {0, 1};
    auto k = xt::fftw::rfftscale<double>(sin.shape()[0], dx);
    xt::xarray<std::complex<double>> sin_derivative_fs = xt::eval(i * k * sin_fs);

    // transform back to normal space
    auto sin_derivative = xt::fftw::irfft(sin_derivative_fs);

    std::cout << "x:              " << x << std::endl;
    std::cout << "sin:            " << sin << std::endl;
    std::cout << "cos:            " << xt::cos(x) << std::endl;
    std::cout << "sin_derivative: " << sin_derivative << std::endl;

**Outputs**

.. code::

    x:              { 0.      ,  0.031416,  0.062832,  0.094248, ...,  6.251769}
    sin:            { 0.000000e+00,  3.141076e-02,  6.279052e-02,  9.410831e-02, ..., -3.141076e-02}
    cos:            { 1.000000e+00,  9.995066e-01,  9.980267e-01,  9.955620e-01, ...,  9.995066e-01}
    sin_derivative: { 1.000000e+00,  9.995066e-01,  9.980267e-01,  9.955620e-01, ...,  9.995066e-01}

xtensor-io
----------

.. image:: xtensor-io.svg
   :alt: xtensor-io

The xtensor-io_ project is an extension to the xtensor library for reading and
writing image, sound and npz file formats to and from xtensor data structures.

xtensor-ros
-----------

.. image:: xtensor-ros.svg
   :alt: xtensor-ros

The xtensor-ros_ project is an extension to the xtensor library providing
helper functions to easily send and receive xtensor and xarray datastructures
as ROS messages.

xsimd
-----

.. image:: xsimd.svg
   :alt: xsimd

The xsimd_ project provides a unified API for making use of the SIMD features
of modern preprocessors for C++ library authors. It also provides accelerated
implementation of common mathematical functions operating on batches.

xsimd_ is an optional dependency to ``xtensor`` which enable SIMD vectorization
of xtensor operations. This feature is enabled with the ``XTENSOR_USE_XSIMD``
compilation flag, which is set to ``false`` by default.

xtl
---

.. image:: xtl.svg
   :alt: xtl

The xtl_ project, the only dependency of ``xtensor`` is a C++ template library
holding the implementation of basic tools used across the libraries in the
QuantStack ecosystem.

.. _xtensor-python: https://github.com/QuantStack/xtensor-python
.. _xtensor-python-cookiecutter: https://github.com/QuantStack/xtensor-python-cookiecutter
.. _xtensor-julia: https://github.com/QuantStack/xtensor-julia
.. _xtensor-julia-cookiecutter: https://github.com/QuantStack/xtensor-julia-cookiecutter
.. _xtensor-r: https://github.com/QuantStack/xtensor-r
.. _xtensor-blas: https://github.com/QuantStack/xtensor-blas
.. _xtensor-io: https://github.com/QuantStack/xtensor-io
.. _xtensor-fftw: https://github.com/egpbos/xtensor-fftw
.. _xtensor-ros: https://github.com/wolfv/xtensor_ros
.. _xsimd: https://github.com/QuantStack/xsimd
.. _xtl: https://github.com/QuantStack/xtl
