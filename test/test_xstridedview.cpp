/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstridedview.hpp"

#include "xtensor/xio.hpp"
#include <iostream>

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

    TEST(xstridedview, diagonal)
    {
        xarray<double> e = xt::arange<double>(1, 10);
        e.reshape({3, 3});
        xarray<double> t = xt::diagonal(e);

        xarray<double> expected = {1, 5, 9};
        ASSERT_EQ(expected, t);

        xt::xarray<double> f = xt::arange(12);
        f.reshape({4, 3});

        xarray<double> exp_1 = {1, 5};
        ASSERT_TRUE(all(equal(exp_1, xt::diagonal(f, 1))));
        xarray<double> exp_2 = {0, 4, 8};
        EXPECT_EQ(exp_2, xt::diagonal(f));
        xarray<double> exp_3 = {3, 7, 11};
        EXPECT_EQ(exp_3, xt::diagonal(f, -1));
        xarray<double> exp_4 = {6, 10};
        EXPECT_EQ(exp_4, xt::diagonal(f, -2));
    }

    TEST(xstridedview, diagonal_advanced)
    {
        xarray<double> e = xt::arange<double>(0, 24);
        e.reshape({2, 2, 2, 3});

        xarray<double> d1 = xt::diagonal(e);

        xarray<double> expected = {{{ 0, 18},
                                    { 1, 19},
                                    { 2, 20}},
                                   {{ 3, 21},
                                    { 4, 22},
                                    { 5, 23}}};
        ASSERT_EQ(expected, d1);

        std::vector<double> d2 = {6, 7, 8, 9, 10, 11};
        xarray<double> expected_2;
        expected_2.reshape({2, 3, 1});
        std::copy(d2.begin(), d2.end(), expected_2.data().begin());

        xarray<double> t2 = xt::diagonal(e, 1);
        ASSERT_EQ(expected_2, t2);

        std::vector<double> d3 = {3, 9, 15, 21};
        xarray<double> expected_3;
        expected_3.reshape({2, 2, 1});
        std::copy(d3.begin(), d3.end(), expected_3.data().begin());
        xarray<double> t3 = xt::diagonal(e, -1, 2, 3);
        ASSERT_EQ(expected_3, t3);

    }

    TEST(xstridedview, flipud)
    {
        xarray<double> e = xt::arange<double>(1, 10);
        e.reshape({3, 3});
        xarray<double> t = xt::flip(e, 0);
        xarray<double> expected = {{7,8,9},{4,5,6},{1,2,3}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(7, t[idx]);
        ASSERT_EQ(2, t(2, 1));
        ASSERT_EQ(7, t.element(idx.begin(), idx.end()));

        xarray<double> f = xt::arange<double>(12);
        f.reshape({2, 2, 3});
        xarray<double> ft = xt::flip(f, 0);
        xarray<double> expected_2 = {{{ 6,  7,  8},
                                      { 9, 10, 11}},
                                     {{ 0,  1,  2},
                                      { 3,  4,  5}}};
        ASSERT_EQ(expected_2, ft);
    }

    TEST(xstridedview, fliplr)
    {
        xarray<double> e = xt::arange<double>(1, 10);
        e.reshape({3, 3});
        xarray<double> t = xt::flip(e, 1);
        xarray<double> expected = {{3,2,1},{6,5,4},{9,8,7}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(3, t[idx]);
        ASSERT_EQ(8, t(2, 1));
        ASSERT_EQ(3, t.element(idx.begin(), idx.end()));

        xarray<double> f = xt::arange<double>(12);
        f.reshape({2, 2, 3});
        xarray<double> ft = xt::flip(f, 1);
        xarray<double> expected_2 = {{{  3,  4,  5},
                                      {  0,  1,  2}},
                                     {{  9, 10, 11},
                                      {  6,  7,  8}}};

        ASSERT_EQ(expected_2, ft);
        // TODO currently doesn't work on non-strided xexpressions
        // auto flipped_range = xt::flip(xt::stack(xt::xtuple(arange<double>(2), arange<double>(2))), 1);
        // xarray<double> expected_range = {{1, 0}, {1, 0}};
        // ASSERT_TRUE(all(equal(flipped_range, expected_range)));
    }
}