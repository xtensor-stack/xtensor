.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _histogram:

Histogram
=========

Basic usage
-----------

.. note::

    .. code-block:: cpp

        xt::histogram(a, bins[, weights][, density])
        xt::histogram_bin_edges(a[, weights][, left, right][, bins][, mode])

    Any of the options ``[...]`` can be omitted (though the order must be preserved). The defaults are:

    *   ``weights = xt::ones(data.shape())``
    *   ``density = false``
    *   ``left = xt::amin(data)(0)``
    *   ``right = xt::amax(data)(0)``
    *   ``bins = 10``
    *   ``mode = xt::histogram::automatic``

The behavior, in-, and output of ``histogram`` is similar to that of :any:`numpy.histogram` with that difference that the bin-edges are obtained by a separate function call:

.. code-block:: cpp

    #include <xtensor/xtensor.hpp>
    #include <xtensor/xhistogram.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xtensor<double,1> data = {1., 1., 2., 2., 3.};

        xt::xtensor<double,1> count = xt::histogram(data, std::size_t(2));

        xt::xtensor<double,1> bin_edges = xt::histogram_bin_edges(data, std::size_t(2));

        return 0;
    }

Bin-edges algorithm
-------------------

To customize the algorithm to be used to construct the histogram, one needs to make use of the latter ``histogram_bin_edges``. For example:

.. code-block:: cpp

    #include <xtensor/xtensor.hpp>
    #include <xtensor/xhistogram.hpp>
    #include <xtensor/xio.hpp>

    int main()
    {
        xt::xtensor<double,1> data = {1., 1., 2., 2., 3.};

        xt::xtensor<double,1> bin_edges = xt::histogram_bin_edges(data, std::size_t(2), xt::histogram_algorithm::uniform);

        xt::xtensor<double,1> prob = xt::histogram(data, bin_edges, true);

        std::cout << bin_edges << std::endl;
        std::cout << prob << std::endl;

        return 0;
    }

The following algorithms are available:

*   ``automatic``: equivalent to ``linspace``.

*   ``linspace``: linearly spaced bin-edges.

*   ``logspace``: bins that logarithmically increase in size.

*   ``uniform``: bin-edges such that the number of data points is the same in all bins (as much as possible).
