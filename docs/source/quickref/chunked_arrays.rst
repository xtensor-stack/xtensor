.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Chunked arrays
==============

Motivation
----------

Arrays can be very large and may not fit in memory. In this case, you may not be
able to use an in-memory array such as an ``xarray``. A solution to this problem
is to cut up the large array into many small arrays, called chunks. Not only do
the chunks fit comfortably in memory, but this also allows to process them in
parallel, including in a distributed environment (although this is not supported
yet).

Formats for the storage of arrays such as `Zarr <https://zarr.readthedocs.io>`_
specifically target chunked arrays. Such formats are becoming increasingly
popular in the field of big data, since the chunks can be stored in the cloud.

In-memory chunked arrays
------------------------

This may not look very useful at first sight, since each chunk (and thus the
whole array) is hold in memory. It means that it cannot work with very large
arrays, but it may be used to parallelize an algorithm, by processing several
chunks at the same time.

An in-memory chunked array has the following type:

.. code::

    #include "xtensor/xchunked_array.hpp"

    using data_type = double;
    // don't use this code:
    using inmemory_chunked_array = xt::xchunked_array<xarray<xarray<data_type>>>;

But you should not directly use this type to create a chunked array. Instead,
use the `chunked_array` factory function:

.. code::

    #include "xtensor/xchunked_array.hpp"

    std::vector<std::size_t> shape = {10, 10, 10};
    std::vector<std::size_t> chunk_shape = {2, 3, 4};
    auto a = xt::chunked_array<double>(shape, chunk_shape);
    // a is an in-memory chunked array
    // each chunk is an xarray<double>, and chunks are hold in an xarray
    // thus a is an xarray of xarray<double> elements
    a(3, 9, 2) = 1.;  // this will address the chunk of index (1, 3, 0)
                      // and in this chunk, the element of index (1, 0, 2)

Chunked arrays implement the full semantic of ``xarray``, including lazy
evaluation.

Stored chunked arrays
---------------------

These are arrays whose chunks are stored on a file system, allowing for
persistence of data. In particular, they are used as a building block for the
`xtensor-zarr <https://github.com/xtensor-stack/xtensor-zarr>`_ library.

For further dedails, please refer to the documentation
of `xtensor-io <https://xtensor-io.readthedocs.io>`_.
