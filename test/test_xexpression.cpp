/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    auto fun()
    {
        auto sa = make_xshared(xarray<double>({{1,2,3,4}, {5,6,7,8}}));
        return sa + sa * sa - sa;
    }

    TEST(xexpression, shared_basic)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> ca = {{1,2,3,4}, {5,6,7,8}};

        auto sa = make_xshared(std::move(a));

        EXPECT_EQ(sa.dimension(), 2);
        EXPECT_EQ(sa.shape(), ca.shape());
        EXPECT_EQ(sa.strides(), ca.strides());
        EXPECT_EQ(sa(1, 3), ca(1, 3));
        EXPECT_EQ(sa.storage(), ca.storage());
        EXPECT_EQ(sa.data_offset(), ca.data_offset());
        EXPECT_EQ(sa.data()[0], ca.data()[0]);
        layout_type L = decltype(sa)::static_layout;
        bool contig = decltype(sa)::contiguous_layout;
        EXPECT_EQ(L, XTENSOR_DEFAULT_LAYOUT);
        EXPECT_EQ(contig, true);

        EXPECT_EQ(sa.use_count(), 1);
        auto cpysa = sa;
        EXPECT_EQ(sa.use_count(), 2);
    }

    TEST(xexpression, shared_xfunctions)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> ca = {{1,2,3,4}, {5,6,7,8}};

        auto sa = make_xshared(std::move(a));

        auto expr1 = sa + sa;
        auto expr2 = a + a;

        EXPECT_EQ(sa.use_count(), 3);
        EXPECT_TRUE(all(equal(expr1, expr2)));
    }

    TEST(xexpression, shared_expr_return)
    {
        auto expr = fun();
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        EXPECT_EQ(expr, a * a);
    }
}  // namespace xt
