/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#if (defined(__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "xtensor/xrandom.hpp"
#pragma GCC diagnostic pop
#else
#include "xtensor/xrandom.hpp"
#endif
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

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
        auto ac1 = xt::random::choice(a, 5, false);
        auto ac2 = xt::random::choice(a, 5, false);
        xt::random::seed(42);
        auto ac3 = xt::random::choice(a, 5, false);
        ASSERT_EQ(ac1, ac3);
        ASSERT_NE(ac1, ac2);

        xt::random::seed(42);
        auto acr1 = xt::random::choice(a, 5, true);
        auto acr2 = xt::random::choice(a, 5, true);
        xt::random::seed(42);
        auto acr3 = xt::random::choice(a, 5, true);
        ASSERT_EQ(acr1, acr3);
        ASSERT_NE(acr1, acr2);

        xarray<double> b = {-1, 1};
        xt::random::seed(42);
        ASSERT_THROW(xt::random::choice(b, 5, false), std::runtime_error);
        ASSERT_NO_THROW(xt::random::choice(b, 5, true));
        xarray<double> multidim_input = { {1,2,3}, {3,4,5} };
        ASSERT_THROW(xt::random::choice(multidim_input, 5, true), std::runtime_error);
    }

    TEST(xrandom, shuffle)
    {
        xarray<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        xt::random::shuffle(a);
        EXPECT_FALSE(std::is_sorted(a.begin(), a.end()));

        xarray<double> b = a;
        b.resize({b.size(), 1});
        xt::random::seed(42);
        xt::random::shuffle(a);
        xt::random::seed(42);
        xt::random::shuffle(b);
        b.resize({b.size()});
        EXPECT_EQ(a, b);

        a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        a.reshape({3, 4});
        auto ar = a;

        xt::random::seed(123);
        // Unfortunately MSVC & OS X seem to produce different shuffles even though the
        // generated integer sequence should be the same ...
#ifdef __linux__
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(0, 1, 2)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(1, 2, 0)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(0, 2, 1)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(0, 1, 2)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(1, 0, 2)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(1, 2, 0)), a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(2, 1, 0)), a);
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        EXPECT_EQ(xt::view(ar, keep(0, 2, 1)), a);
#else
        xt::random::shuffle(a);
        xt::random::shuffle(a);
        EXPECT_FALSE(std::is_sorted(a.begin(), a.end()));
#endif

    }

    TEST(xrandom, permutation)
    {
        xt::random::seed(123);
        auto r1 = xt::random::permutation(12);
        xt::random::seed(123);
        xtensor<int, 1> a1 = arange<int>(12);
        xtensor<int, 1> ac1 = a1;
        xt::random::shuffle(a1);
        EXPECT_EQ(a1, r1);
        EXPECT_NE(r1, ac1);

        xt::random::seed(123);
        auto r2 = xt::random::permutation(ac1);
        EXPECT_EQ(a1, r2);
    }
}
