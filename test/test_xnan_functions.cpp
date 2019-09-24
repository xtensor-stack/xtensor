/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrided_view.hpp"

namespace xt
{
    using namespace std::complex_literals;
    static const double nanv = std::nan("0");
    namespace nantest
    {
        xarray<double> aN = {{ nanv, nanv, 123, 3 }, { 1, 2, nanv, 3 }, { 1, 1, nanv, 3 }};
        xarray<double> aR = {{ 0, 0, 123 , 3 }, { 1, 2, 0 , 3}, { 1, 1, 0, 3 }};
        xarray<double> aP = {{ 1, 1, 123 , 3}, { 1, 2, 1, 3 }, { 1, 1, 1, 3 }};

        xarray<double> xN = {{{nanv, nanv}, {1,2}}, {{3, nanv}, {nanv, 5}}};
        xarray<double> xR = {{{0, 0}, {1,2}}, {{3, 0}, {0, 5}}};
        xarray<double> xP = {{{1, 1}, {1,2}}, {{3, 1}, {1, 5}}};

        xarray<std::complex<double>> cN = {
            {1.0 + 1.0i, 1.0 + 1.0i, nanv      },
            {1.0 - 1.0i, 1.0       , 3.0 + 2.0i}
        };

    }

    TEST(xnanfunctions, count_nonnan)
    {
        xarray<double> a = {{0, 1, 2, 3}, {nanv, nanv, nanv, nanv}, {3, nanv, 1, nanv}};
        std::size_t as = count_nonnan(a)();
        std::size_t ase = count_nonnan(a, evaluation_strategy::immediate)();
        EXPECT_EQ(as, 6u);
        EXPECT_EQ(ase, 6u);

        xarray<std::size_t> ea0 = {2, 1, 2, 1};
        xarray<std::size_t> ea1 = {4, 0, 2};

        EXPECT_EQ(count_nonnan(a, {0}), ea0);
        EXPECT_EQ(count_nonnan(a, {1}), ea1);

        EXPECT_EQ(count_nonnan(a, {0}, evaluation_strategy::immediate), ea0);
        EXPECT_EQ(count_nonnan(a, {1}, evaluation_strategy::immediate), ea1);
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
        EXPECT_EQ(res(0), 137);

        EXPECT_EQ(nansum(nantest::aN, {0}), sum(nantest::aR, {0}));
        EXPECT_EQ(nansum(nantest::aN, {1}), sum(nantest::aR, {1}));
        EXPECT_EQ(nansum(nantest::xN, {0}), sum(nantest::xR, {0}));
        EXPECT_EQ(nansum(nantest::xN, {1}), sum(nantest::xR, {1}));
        EXPECT_EQ(nansum(nantest::xN, {2}), sum(nantest::xR, {2}));
    }

    TEST(xnanfunctions, nanprod)
    {
        xarray<double> res = nanprod(nantest::aN);
        xarray<double> res0 = prod(nantest::aP);
        EXPECT_EQ(res(0), 6642);

        EXPECT_EQ(nanprod(nantest::aN, {0}), prod(nantest::aP, {0}));
        EXPECT_EQ(nanprod(nantest::aN, {1}), prod(nantest::aP, {1}));
        EXPECT_EQ(nanprod(nantest::xN, {0}), prod(nantest::xP, {0}));
        EXPECT_EQ(nanprod(nantest::xN, {1}), prod(nantest::xP, {1}));
        EXPECT_EQ(nanprod(nantest::xN, {2}), prod(nantest::xP, {2}));
    }

    TEST(xnanfunctions, nancumsum)
    {
        EXPECT_EQ(nancumsum(nantest::aN), cumsum(nantest::aR));
        EXPECT_EQ(nancumsum(nantest::aN, 0), cumsum(nantest::aR, 0));
        EXPECT_EQ(nancumsum(nantest::aN, 1), cumsum(nantest::aR, 1));
        EXPECT_EQ(nancumsum(nantest::xN, 0), cumsum(nantest::xR, 0));
        EXPECT_EQ(nancumsum(nantest::xN, 1), cumsum(nantest::xR, 1));
        EXPECT_EQ(nancumsum(nantest::xN, 2), cumsum(nantest::xR, 2));
    }

    TEST(xnanfunctions, multid)
    {
        xarray<double> arr = xt::arange(3 * 4 * 2 * 8 * 7);
        arr.reshape({3, 4, 2, 8, 7});
        xarray<double> carr = arr;
        strided_view(arr, {0, xt::ellipsis()}) = nanv;
        strided_view(carr, {0, xt::ellipsis()}) = 0;

        EXPECT_EQ(nancumsum(arr, 0), cumsum(carr, 0));
        EXPECT_EQ(nancumsum(arr, 1), cumsum(carr, 1));
        EXPECT_EQ(nancumsum(arr, 2), cumsum(carr, 2));
        EXPECT_EQ(nancumsum(arr, 3), cumsum(carr, 3));
        EXPECT_EQ(nancumsum(arr, 4), cumsum(carr, 4));
    }

    TEST(xnanfunctions, nancumprod)
    {
        EXPECT_EQ(nancumprod(nantest::aN), cumprod(nantest::aP));
        EXPECT_EQ(nancumprod(nantest::aN, 0), cumprod(nantest::aP, 0));
        EXPECT_EQ(nancumprod(nantest::aN, 1), cumprod(nantest::aP, 1));
        EXPECT_EQ(nancumprod(nantest::xN, 0), cumprod(nantest::xP, 0));
        EXPECT_EQ(nancumprod(nantest::xN, 1), cumprod(nantest::xP, 1));
        EXPECT_EQ(nancumprod(nantest::xN, 2), cumprod(nantest::xP, 2));
    }

    TEST(xnanfunctions, nanmean)
    {
        auto as = nanmean(nantest::aN)();
        auto ase = nanmean(nantest::aN, evaluation_strategy::immediate)();
        EXPECT_EQ(as, 17.125);
        EXPECT_EQ(ase, 17.125);

        xarray<double> eaN0 = {1.0, 1.5, 123, 3};
        xarray<double> eaN1 = {63.0, 2.0, 5.0/3.0};

        EXPECT_EQ(nanmean(nantest::aN, {0}), eaN0);
        EXPECT_EQ(nanmean(nantest::aN, {1}), eaN1);

        std::array<std::size_t, 1> axis{0};
        EXPECT_EQ(nanmean(nantest::aN, axis), eaN0);

        EXPECT_EQ(nanmean(nantest::aN, {0}, evaluation_strategy::immediate), eaN0);
        EXPECT_EQ(nanmean(nantest::aN, {1}, evaluation_strategy::immediate), eaN1);

        auto cs = nanmean(nantest::cN)();
        auto cse = nanmean(nantest::cN, evaluation_strategy::immediate)();
        EXPECT_EQ(cs, std::complex<double>(1.4, 0.6));
        EXPECT_EQ(cse, std::complex<double>(1.4, 0.6));

        xarray<std::complex<double>> ecN0 = {1.0 + 0.0i, 1.0+0.5i, 3.0+2.0i};
        xarray<std::complex<double>> ecN1 = {1.0 + 1.0i, (5.0 + 1.0i) / 3.0};

        EXPECT_EQ(nanmean(nantest::cN, {0}), ecN0);
        EXPECT_EQ(nanmean(nantest::cN, {1}), ecN1);

        EXPECT_EQ(nanmean(nantest::cN, {0}, evaluation_strategy::immediate), ecN0);
        EXPECT_EQ(nanmean(nantest::cN, {1}, evaluation_strategy::immediate), ecN1);
    }


    TEST(xnanfunctions, nanvar)
    {
        auto as = nanvar(nantest::aN)();
        auto ase = nanvar(nantest::aN, evaluation_strategy::immediate)();
        EXPECT_EQ(as, 1602.109375);
        EXPECT_EQ(ase, 1602.109375);

        xarray<double> eaN0 = {0.0, 0.25, 0.0, 0.0};
        xarray<double> eaN1 = {3600.0, 2.0/3.0, 8.0/9.0};

        EXPECT_EQ(nanvar(nantest::aN, {0}), eaN0);
        EXPECT_TRUE(allclose(nanvar(nantest::aN, {1}), eaN1));

        std::array<std::size_t, 1> axis{0};
        EXPECT_EQ(nanvar(nantest::aN, axis), eaN0);

        EXPECT_EQ(nanvar(nantest::aN, {0}, evaluation_strategy::immediate), eaN0);
        EXPECT_TRUE(allclose(nanvar(nantest::aN, {1}, evaluation_strategy::immediate), eaN1));
    }
}
