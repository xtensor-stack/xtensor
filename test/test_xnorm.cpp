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
    TEST(xnorm, norm_traits)
    {
        EXPECT_TRUE((std::is_same<norm_type_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<uint8_t>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<int>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_type_t<std::vector<long double>>, long double>::value));

        EXPECT_TRUE((std::is_same<squared_norm_type_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<uint8_t>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<int>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<squared_norm_type_t<std::vector<long double>>, long double>::value));
    }

    TEST(xnorm, scalar)
    {
        EXPECT_EQ(norm_l0(2), 1u);
        EXPECT_EQ(norm_l0(-2), 1u);
        EXPECT_EQ(norm_l0(0), 0u);
        EXPECT_EQ(norm_l0(2.0), 1u);
        EXPECT_EQ(norm_l0(-2.0), 1u);
        EXPECT_EQ(norm_l0(0.0), 0u);

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
        std::complex<double> c{3.0, -4.0};

        EXPECT_EQ(norm_l0(c), 1u);
        EXPECT_EQ(norm_lp(c, 0), 1.0);

        EXPECT_EQ(norm_l1(c), 7.0);
        EXPECT_EQ(norm_lp(c, 1), 7.0);

        EXPECT_EQ(norm_sq(c), 25.0);
        EXPECT_EQ(norm_lp_to_p(c, 2), 25.0);
        EXPECT_EQ(norm_sq(c), std::norm(c));

        EXPECT_EQ(norm_l2(c), 5.0);
        EXPECT_EQ(norm_lp(c, 2), 5.0);
        EXPECT_EQ(norm_l2(c), std::abs(c));

        EXPECT_EQ(norm_linf(c), 4.0);
    }

    TEST(xnorm, scalar_array)
    {
        xarray<int> a = -ones<int>({9});

        EXPECT_EQ(norm_l0(a)(), 9u);
        EXPECT_EQ(norm_lp_to_p(a, 0.0)(), 9.0);
        EXPECT_EQ(norm_l1(a)(), 9);
        EXPECT_EQ(norm_lp(a, 1.0)(), 9.0);
        EXPECT_EQ(norm_sq(a)(), 9);
        EXPECT_EQ(norm_l2(a)(), 3.0);
        EXPECT_EQ(norm_lp(a, 2.0)(), 3.0);
        EXPECT_EQ(norm_linf(a)(), 1);

        EXPECT_EQ(norm_l0(a, evaluation_strategy::immediate)(), 9u);
        EXPECT_EQ(norm_lp_to_p(a, 0.0, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_l1(a, evaluation_strategy::immediate)(), 9);
        EXPECT_EQ(norm_lp(a, 1.0, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_sq(a, evaluation_strategy::immediate)(), 9);
        EXPECT_EQ(norm_l2(a, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_lp(a, 2.0, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_linf(a, evaluation_strategy::immediate)(), 1);
    }

    TEST(xnorm, complex_array)
    {
        xarray<std::complex<double>> a = -ones<std::complex<double>>({9});

        EXPECT_EQ(norm_l0(a)(), 9u);
        EXPECT_EQ(norm_lp_to_p(a, 0.0)(), 9.0);
        EXPECT_EQ(norm_l1(a)(), 9.0);
        EXPECT_EQ(norm_lp(a, 1.0)(), 9.0);
        EXPECT_EQ(norm_sq(a)(), 9.0);
        EXPECT_EQ(norm_l2(a)(), 3.0);
        EXPECT_EQ(norm_lp(a, 2.0)(), 3.0);
        EXPECT_EQ(norm_linf(a)(), 1.0);

        EXPECT_EQ(norm_l0(a, {0}, evaluation_strategy::immediate)(), 9u);
        EXPECT_EQ(norm_lp_to_p(a, 0.0, {0}, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_l1(a, {0}, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_lp(a, 1.0, {0}, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_sq(a, {0}, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_l2(a, {0}, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_lp(a, 2.0, {0}, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_linf(a, {0}, evaluation_strategy::immediate)(), 1.0);

        EXPECT_EQ(norm_l0(a, {0}, evaluation_strategy::immediate)(), 9u);
        EXPECT_EQ(norm_lp_to_p(a, 0.0, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_l1(a, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_lp(a, 1.0, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_sq(a, evaluation_strategy::immediate)(), 9.0);
        EXPECT_EQ(norm_l2(a, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_lp(a, 2.0, evaluation_strategy::immediate)(), 3.0);
        EXPECT_EQ(norm_linf(a, evaluation_strategy::immediate)(), 1.0);
    }

    TEST(xnorm, matrix)
    {
        xarray<double> a = {{ -1.0, 2.0},
                            { -3.0, 4.0}};

        EXPECT_EQ(norm_l0(a)(), 4u);
        EXPECT_EQ(norm_l1(a)(), 10.0);
        EXPECT_EQ(norm_sq(a)(), 30.0);
        EXPECT_EQ(norm_linf(a)(), 4.0);
        EXPECT_EQ(norm_induced_l1(a)(), 6.0);
        EXPECT_EQ(norm_induced_linf(a)(), 7.0);

        EXPECT_EQ(norm_l0(a, evaluation_strategy::immediate)(), 4u);
        EXPECT_EQ(norm_l1(a, evaluation_strategy::immediate)(), 10.0);
        EXPECT_EQ(norm_sq(a, evaluation_strategy::immediate)(), 30.0);
        EXPECT_EQ(norm_linf(a, evaluation_strategy::immediate)(), 4.0);
        EXPECT_EQ(norm_induced_l1(a, evaluation_strategy::immediate)(), 6.0);
        EXPECT_EQ(norm_induced_linf(a, evaluation_strategy::immediate)(), 7.0);
    }
}  // namespace xt
