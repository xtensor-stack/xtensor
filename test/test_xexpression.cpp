/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include <sstream>

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"

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
        
        std::stringstream buffer;
        buffer << sa;
        EXPECT_EQ(buffer.str(), "{{ 1.,  2.,  3.,  4.},\n { 5.,  6.,  7.,  8.}}");
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
        std::stringstream buffer;
        buffer << expr1;
        EXPECT_EQ(buffer.str(), "{{  2.,   4.,   6.,   8.},\n { 10.,  12.,  14.,  16.}}");

        // Compilation test
        auto sexpr1 = make_xshared(std::move(expr1));
        using expr_type = decltype(sexpr1);
        using strides_type = typename expr_type::strides_type;
        using inner_strides_type = typename expr_type::inner_strides_type;
        using backstrides_type = typename expr_type::backstrides_type;
        using inner_strides_tybackstrides_typepe = typename expr_type::inner_backstrides_type;
    }

    TEST(xexpression, shared_expr_return)
    {
        auto expr = fun();
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        EXPECT_EQ(expr, a * a);
    }
}  // namespace xt
