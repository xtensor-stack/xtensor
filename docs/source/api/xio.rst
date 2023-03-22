.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xio: pretty printing
====================

Defined in ``xtensor/xio.hpp``

This file defines functions for pretty printing xexpressions. It defines appropriate
overloads for the ``<<`` operator for std::ostreams and xexpressions.

.. code::

    #include <xtensor/xio.hpp>
    #include <xtensor/xarray.hpp>

    int main()
    {
        xt::xarray<double> a = {{1,2,3}, {4,5,6}};
        std::cout << a << std::endl;
        return 0;
    }

Will print

.. code::

    {{ 1., 2., 3.},
     { 4., 5., 6.}}

With the following functions, the global print options can be set:

.. doxygenfunction:: xt::print_options::set_line_width

.. doxygenfunction:: xt::print_options::set_threshold

.. doxygenfunction:: xt::print_options::set_edge_items

.. doxygenfunction:: xt::print_options::set_precision

On can also locally overwrite the print options with io manipulators:

.. doxygenclass:: xt::print_options::line_width

.. doxygenclass:: xt::print_options::threshold

.. doxygenclass:: xt::print_options::edge_items

.. doxygenclass:: xt::print_options::precision
