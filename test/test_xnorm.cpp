/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xnorm.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"

#include <limits>

namespace xt
{
    TEST(xnorm, scalar)
    {
        EXPECT_EQ(norm_l0(2), 1);
        EXPECT_EQ(norm_l0(-2), 1);
        EXPECT_EQ(norm_l0(0), 0);
        EXPECT_EQ(norm_l0(2.0), 1);
        EXPECT_EQ(norm_l0(-2.0), 1);
        EXPECT_EQ(norm_l0(0.0), 0);

        EXPECT_EQ(norm_l1(2), 2);
        EXPECT_EQ(norm_l1(-2), 2);
        EXPECT_EQ(norm_l1(2.0), 2.0);
        EXPECT_EQ(norm_l1(-2.0), 2.0);

        EXPECT_EQ(norm_l2(2), 2);
        EXPECT_EQ(norm_l2(-2), 2);
        EXPECT_EQ(norm_l2(2.0), 2.0);
        EXPECT_EQ(norm_l2(-2.0), 2.0);

        EXPECT_EQ(norm_linf(2), 2);
        EXPECT_EQ(norm_linf(-2), 2);
        EXPECT_EQ(norm_linf(2.0), 2.0);
        EXPECT_EQ(norm_linf(-2.0), 2.0);

        EXPECT_EQ(norm_sq(2), 4);
        EXPECT_EQ(norm_sq(-2), 4);
        EXPECT_EQ(norm_sq(2.5), 6.25);
        EXPECT_EQ(norm_sq(-2.5), 6.25);

        EXPECT_EQ(norm_lp(0, 0), 0);
        EXPECT_EQ(norm_lp(-2, 0), 1);
        EXPECT_EQ(norm_lp(0, 1), 0);
        EXPECT_EQ(norm_lp(-2, 1), 2);
    }

    TEST(xnorm, complex)
    {
        std::complex<double> c{ 3.0, 4.0 };
        EXPECT_EQ(norm_sq(c), 25.0);
        EXPECT_EQ(norm_sq(c), std::norm(c));

        EXPECT_EQ(norm_l2(c), 5.0);
        EXPECT_EQ(norm_l2(c), std::abs(c));
    }

    TEST(xnorm, scalar_array)
    {
        xarray<int> i1 = -ones<int>({9});

        EXPECT_EQ(norm_l0(i1)(), 9);
        EXPECT_EQ(norm_l1(i1)(), 9);
        EXPECT_EQ(norm_lp(i1, 1.0)(), 9);
        EXPECT_EQ(norm_sq(i1)(), 9);
        EXPECT_EQ(norm_l2(i1)(), 3);
        EXPECT_EQ(norm_lp(i1, 2.0)(), 3);
        EXPECT_EQ(norm_linf(i1)(), 1);
    }

    TEST(xnorm, complex_array)
    {
        xarray<std::complex<double>> i1 = -ones<std::complex<double>>({9});

        EXPECT_EQ(norm_l0(i1)(), 9);
        EXPECT_EQ(norm_sq(i1)(), 9.0);
        EXPECT_EQ(norm_l2(i1)(), 3.0);
    }
} // namespace xt