/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xrandom, random)
    {
        auto r = random::rand<double>({3, 3});
        xarray<double> a = r;
        xarray<double> b = r;
        xarray<double> c = r;

        ASSERT_NE(a(0, 0), a(0, 1));
        ASSERT_NE(a, b);
        ASSERT_NE(a, c);

        xarray<double> other_rand = random::rand<double>({3, 3});
        ASSERT_NE(a, other_rand);

        random::seed(0);
        auto same_d_a = random::rand<double>({3, 3});
        xarray<double> same_a = same_d_a;

        random::seed(0);
        auto same_d_b = random::rand<double>({3, 3});
        xarray<double> same_b = same_d_b;

        ASSERT_EQ(same_a, same_b);

        // check that it compiles
        xarray<int> q = random::randint<int>({3, 3});

        // checking if internal state needs reset
        auto n_dist = random::randn<double>({3, 3});
        xarray<double> p1 = n_dist;
        xarray<double> p2 = n_dist;
        xarray<double> p3 = n_dist;
        ASSERT_NE(p1, p2);
        ASSERT_NE(p1, p3);
    }

    TEST(xrandom, choice)
    {
        xarray<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        xt::random::seed(42);
        auto ac1 = xt::random::choice(a, 5);
        auto ac2 = xt::random::choice(a, 5);
        xt::random::seed(42);
        auto ac3 = xt::random::choice(a, 5);
        EXPECT_EQ(ac1, ac3);
        EXPECT_NE(ac1, ac2);
    }
}
