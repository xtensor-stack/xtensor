.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xhistogram
==========

Defined in ``xtensor/xhistogram.hpp``

.. doxygenenum:: xt::histogram_algorithm

.. doxygenfunction:: xt::histogram(E1&&, E2&&, E3&&, bool)

.. doxygenfunction:: xt::bincount(E1&&, E2&&, std::size_t)

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2&&, E3, E3, std::size_t, histogram_algorithm)

.. doxygenfunction:: xt::digitize(E1&&, E2&&, E3&&, bool, bool)

.. doxygenfunction:: xt::bin_items(size_t, E&&)

Further overloads
-----------------

.. doxygenfunction:: xt::histogram(E1&&, E2&&, bool)

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, bool)

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, E2, E2, bool)

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, E2&&, bool)

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, E2&&, E3, E3, bool)

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2, E2, std::size_t, histogram_algorithm)

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2&&, std::size_t, histogram_algorithm)

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, std::size_t, histogram_algorithm)

.. doxygenfunction:: xt::bin_items(size_t, size_t)
