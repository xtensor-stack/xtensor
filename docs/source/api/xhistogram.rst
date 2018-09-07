.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xhistogram
==========

Defined in ``xtensor/xhistogram.hpp``

.. doxygenenum:: xt::histogram_algorithm
   :project: xtensor

.. doxygenfunction:: xt::histogram(E1&&, E2&&, E3&&, bool)
   :project: xtensor

.. doxygenfunction:: xt::bincount(E1&&, E2&&, std::size_t)
   :project: xtensor

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2&&, E3, E3, std::size_t, histogram_algorithm)
   :project: xtensor

Further overloads
-----------------

.. doxygenfunction:: xt::histogram(E1&&, E2&&, bool)
   :project: xtensor

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, bool)
   :project: xtensor

.. doxygenfunction:: xt::histogram(E1&&, std::size_t, E2&&, bool)
   :project: xtensor

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2, E2, std::size_t, histogram_algorithm)
   :project: xtensor

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, E2&&, std::size_t, histogram_algorithm)
   :project: xtensor

.. doxygenfunction:: xt::histogram_bin_edges(E1&&, std::size_t, histogram_algorithm)
   :project: xtensor
