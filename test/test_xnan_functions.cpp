/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    static const double nanv = std::nan("0");
    namespace nantest
    {
        xarray<double> aN = {{ nanv, nanv, 123 }, { 1, 2, nanv }, { 1, 1, nanv }};
        xarray<double> aR = {{ 0, 0, 123 }, { 1, 2, 0 }, { 1, 1, 0 }};        
        xarray<double> aP = {{ 1, 1, 123 }, { 1, 2, 1 }, { 1, 1, 1 }};        
    }

    TEST(xnanfunctions, nan_to_num)
    {
        double neg_inf = -std::numeric_limits<double>::infinity();
        double inf = std::numeric_limits<double>::infinity();
        xarray<double> a = {{ nanv, nanv, 123}, {0.5123, neg_inf, inf}};

        auto expr = nan_to_num(a);
        EXPECT_EQ(expr(0, 0), 0);
        EXPECT_EQ(expr(0, 1), 0);
        EXPECT_EQ(expr(0, 2), 123);
        EXPECT_EQ(expr(1, 0), 0.5123);
        EXPECT_EQ(expr(1, 1), std::numeric_limits<double>::lowest());
        EXPECT_TRUE(expr(1, 1) < 0);
        EXPECT_EQ(expr(1, 2), std::numeric_limits<double>::max());

        xarray<double> exp = {{ 0, 0, 123 }, {0.5123, std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max() }};
        xarray<double> assigned = exp;
        EXPECT_EQ(assigned, exp);
    }

    TEST(xnanfunctions, nansum)
    {
        xarray<double> res = nansum(nantest::aN);
        xarray<double> res0 = sum(nantest::aR);
        EXPECT_EQ(res(0), 128);

        EXPECT_EQ(nansum(nantest::aN, {0}), sum(nantest::aR, {0}));
        EXPECT_EQ(nansum(nantest::aN, {1}), sum(nantest::aR, {1}));
    }

    TEST(xnanfunctions, nanprod)
    {
        xarray<double> res = nanprod(nantest::aN);
        xarray<double> res0 = prod(nantest::aP);
        EXPECT_EQ(res(0), 246);

        EXPECT_EQ(nanprod(nantest::aN, {0}), prod(nantest::aP, {0}));
        EXPECT_EQ(nanprod(nantest::aN, {1}), prod(nantest::aP, {1}));
    }

    TEST(xnanfunctions, nancumsum)
    {
        EXPECT_EQ(nancumsum(nantest::aN), cumsum(nantest::aR));
        EXPECT_EQ(nancumsum(nantest::aN, 0), cumsum(nantest::aR, 0));
        EXPECT_EQ(nancumsum(nantest::aN, 1), cumsum(nantest::aR, 1));
    }

    TEST(xnanfunctions, nancumprod)
    {
        EXPECT_EQ(nancumprod(nantest::aN), cumprod(nantest::aP));
        EXPECT_EQ(nancumprod(nantest::aN, 0), cumprod(nantest::aP, 0));
        EXPECT_EQ(nancumprod(nantest::aN, 1), cumprod(nantest::aP, 1));
    }

}
