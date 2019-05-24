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
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xnobroadcast.hpp"

namespace xt
{
	TEST(xnobroadcast, assign)
	{
		xarray<int> a = { 1, 2 , 3 };
		xarray<int> b = { 4, 5 , 6 };

        xarray<int> arrays_sum = a + b;
        xarray<int> br_arrays_sum;
        nobroadcast(br_arrays_sum) = a + b;

        EXPECT_EQ(br_arrays_sum, arrays_sum);

        xscalar<int> k = 3;

        xarray<int> scalar_sum = a + k;
        xarray<int> br_scalar_sum;
        nobroadcast(br_scalar_sum) = a + k;

        EXPECT_EQ(br_scalar_sum, scalar_sum);
	}
    TEST(xnobroadcast, a_plus_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a += a;
        nobroadcast(nobr_a) += nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_minus_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a -= a;
        nobroadcast(nobr_a) -= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_times_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a *= a;
        nobroadcast(nobr_a) *= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_divide_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a /= a;
        nobroadcast(nobr_a) /= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_modulus_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a %= a;
        nobroadcast(nobr_a) %= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_bit_and_assign_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a &= a;
        nobroadcast(nobr_a) &= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_bit_or_assign_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a |= a;
        nobroadcast(nobr_a) |= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }

    TEST(xnobroadcast, a_bit_xor_assign_equal_b)
    {
        xarray<int> a = { 1,2,3 };
        xarray<int> nobr_a;
        nobroadcast(nobr_a) = a;

        a ^= a;
        nobroadcast(nobr_a) ^= nobr_a;

        EXPECT_EQ(nobr_a, a);
    }
}
