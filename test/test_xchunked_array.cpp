/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xbroadcast.hpp"
#include "xtensor/xchunked_array.hpp"
#include "xtensor/xcsv.hpp"

namespace xt
{
    using in_memory_chunked_array = xchunked_array<xarray<xarray<double>>>;

    TEST(xchunked_array, indexed_access)
    {
        std::vector<size_t> shape = {10, 10, 10};
        std::vector<size_t> chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);

        std::vector<size_t> idx = {3, 9, 8};
        double val;

        val = 1.;
        a[idx] = val;
        ASSERT_EQ(a[idx], val);
        ASSERT_EQ(a(3, 9, 8), val);

        val = 2.;
        a(3, 9, 8) = val;
        ASSERT_EQ(a(3, 9, 8), val);
        ASSERT_EQ(a[idx], val);

        val = 3.;
        for (auto& it: a)
            it = val;
        for (auto it: a)
            ASSERT_EQ(it, val);
    }

    TEST(xchunked_array, assign_expression)
    {
        std::vector<size_t> shape1 = {2, 2, 2};
        std::vector<size_t> chunk_shape1 = {2, 3, 4};
        auto a1 = chunked_array<double>(shape1, chunk_shape1);
        double val;

        val = 3.;
        a1 = broadcast(val, a1.shape());
        for (const auto& v: a1)
        {
            EXPECT_EQ(v, val);
        }

        std::vector<size_t> shape2 = {32, 10, 10};
        auto a2 = chunked_array<double>(shape2, chunk_shape1);

        a2 = broadcast(val, a2.shape());
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, val);
        }

        a2 += a2;
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, 2. * val);
        }

        a2 += 2.;
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, 2. * val + 2.);
        }


        xarray<double> a3
          {{1., 2., 3.},
           {4., 5., 6.},
           {7., 8., 9.}};

        EXPECT_EQ(is_chunked(a3), false);

        std::vector<size_t> chunk_shape4 = {2, 2};
        auto a4 = chunked_array(a3, chunk_shape4);

        EXPECT_EQ(is_chunked(a4), true);

        double i = 1.;
        for (const auto& v: a4)
        {
            EXPECT_EQ(v, i);
            i += 1.;
        }

        auto a5 = in_memory_chunked_array(a4);
        EXPECT_EQ(is_chunked(a5), true);
        for (const auto& v: a5.chunk_shape())
        {
            EXPECT_EQ(v, 2);
        }

        auto a6 = chunked_array(a3);
        EXPECT_EQ(is_chunked(a6), true);
        for (const auto& v: a6.chunk_shape())
        {
            EXPECT_EQ(v, 3);
        }
    }
}
