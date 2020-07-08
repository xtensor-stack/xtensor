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

namespace xt
{
    using chunked_array = xt::xchunked_array<xt::xarray<double>>;

    TEST(xchunked_array, indexed_access)
    {
        std::vector<size_t> shape = {10, 10, 10};
        std::vector<size_t> chunk_shape = {2, 3, 4};
        chunked_array a(shape, chunk_shape);

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
        chunked_array a1(shape1, chunk_shape1);
        double val;

        val = 3.;
        a1 = xt::broadcast(val, a1.shape());
        for (const auto& v: a1)
        {
            EXPECT_EQ(v, val);
        }

        std::vector<size_t> shape2 = {32, 10, 10};
        chunked_array a2(shape2, chunk_shape1);

        a2 = xt::broadcast(val, a2.shape());
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, val);
        }

        a2 += a2;
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, 2. * val);
        }
    }
}
