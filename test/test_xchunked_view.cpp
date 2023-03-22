/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/xarray.hpp"
#include "xtensor/xchunked_array.hpp"
#include "xtensor/xchunked_view.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(xchunked_view, iterate)
    {
        std::vector<std::size_t> shape = {3, 4};
        std::vector<std::size_t> chunk_shape = {1, 2};
        xarray<double> a(shape);
        std::size_t chunk_nb = 0;
        auto chunked_view = xchunked_view<xarray<double>>(a, chunk_shape);
        for (auto it = chunked_view.chunk_begin(); it != chunked_view.chunk_end(); it++)
        {
            chunk_nb++;
        }

        std::size_t expected_chunk_nb = (shape[0] / chunk_shape[0]) * (shape[1] / chunk_shape[1]);

        EXPECT_EQ(chunk_nb, expected_chunk_nb);
    }

    TEST(xchunked_view, assign)
    {
        std::vector<std::size_t> shape = {3, 4};
        std::vector<std::size_t> chunk_shape = {1, 2};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.begin());
        xarray<double> b(shape);

        as_chunked(b, chunk_shape) = a;

        EXPECT_EQ(a, b);
    }

    TEST(xchunked_view, assign_chunked_array)
    {
        std::vector<std::size_t> shape = {10, 10, 10};
        std::vector<std::size_t> chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        xarray<double> b(shape);
        auto ref = arange(0, 1000).reshape(shape);

        as_chunked(a, chunk_shape) = ref;
        as_chunked(b, chunk_shape) = a;

        EXPECT_EQ(ref, a);
        EXPECT_EQ(ref, b);
    }
}
