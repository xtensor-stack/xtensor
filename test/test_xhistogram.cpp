/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xhistogram.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    TEST(xhistogram, histogram)
    {
        xt::xtensor<double, 1> data = {1., 1., 2., 2.};

        {
            xt::xtensor<double, 1> count = xt::histogram(data, std::size_t(2));

            EXPECT_EQ(count.size(), std::size_t(2) );
            EXPECT_EQ(count(0), 2.);
            EXPECT_EQ(count(1), 2.);
        }

        {
            xt::xtensor<double, 1> count = xt::histogram(data,
                xt::histogram_bin_edges(data, std::size_t(2), xt::histogram_algorithm::uniform)
            );

            EXPECT_EQ(count.size(), std::size_t(2));
            EXPECT_EQ(count(0), 2.);
            EXPECT_EQ(count(1), 2.);
        }

        {
            xt::xarray<double> arr = {1., 1., 2., 2.};
            xt::xtensor<double, 1> count = xt::histogram(arr, std::size_t(2));

            EXPECT_EQ(count.size(), std::size_t(2) );
            EXPECT_EQ(count(0), 2.);
            EXPECT_EQ(count(1), 2.);
        }
    }

    TEST(xhistogram, bincount)
    {
        xtensor<int, 1> data = {1, 2, 3, 1, 1, 1, 1, 2, 3, 2, 3, 3, 3, 3};
        xtensor<int, 1> weights = xt::ones<int>(data.shape()) * 3;
        xtensor<int, 1> expc = {0, 5, 3, 6};
        auto bc = bincount(data);
        EXPECT_EQ(bc, expc);

        auto bc2 = bincount(data, weights);
        EXPECT_EQ(bc2, expc * 3);

        auto bc3 = bincount(data, 10);
        EXPECT_EQ(bc3.size(), std::size_t(10));
        EXPECT_EQ(bc3(3), expc(3));
    }

    TEST(xhistogram, digitize)
    {
        xt::xtensor<size_t, 1> bin_edges = {0, 10, 20, 30};
        xt::xtensor<size_t, 1> data = {1, 12, 2, 21, 11, 10};
        xt::xtensor<size_t, 1> res_left = {1, 2, 1, 3, 2, 2};
        xt::xtensor<size_t, 1> res_right = {1, 2, 1, 3, 2, 1};
        EXPECT_EQ(xt::digitize(data, bin_edges), res_left);
        EXPECT_EQ(xt::digitize(data, bin_edges, false), res_left);
        EXPECT_EQ(xt::digitize(data, bin_edges, true), res_right);
    }


    TEST(xhistogram, bin_items_1)
    {
        xt::xtensor<size_t, 1> a = xt::bin_items(11, xt::xtensor<double, 1>{0.9, 0.0, 0.0, 0.1});
        xt::xtensor<size_t, 1> b = {9, 0, 0, 2};
        EXPECT_EQ(a, b);
    }

    TEST(xhistogram, bin_items_2)
    {
        xt::xtensor<size_t, 1> a = xt::bin_items(11, xt::xtensor<double, 1>{0.25, 0.25, 0.25, 0.25});
        xt::xtensor<size_t, 1> b = {3, 3, 3, 2};
        EXPECT_EQ(a, b);
    }
}
