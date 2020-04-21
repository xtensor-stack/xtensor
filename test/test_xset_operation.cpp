/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <cstddef>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xset_operation.hpp"

namespace xt
{
    TEST(xset_operation, isin)
    {
        xt::xtensor<int,2> a = {{1, 2, 1}, {0, 3, 1}};
        xt::xtensor<int,1> b = {1, 2};
        xt::xtensor<bool,2> res = {{true, true, true}, {false, false, true}};
        EXPECT_EQ(xt::isin(a, b), res);
        EXPECT_EQ(xt::isin(a, b.begin(), b.end()), res);
        EXPECT_EQ(xt::isin(a, {1, 2}), res);
    }

    TEST(xset_operation, in1d)
    {
        xt::xtensor<int,1> a = {1, 2, 1, 0, 3, 5, 1};
        xt::xtensor<int,1> b = {1, 2};
        xt::xtensor<bool,1> res = {true, true, true, false, false, false, true};
        EXPECT_EQ(xt::in1d(a, b), res);
        EXPECT_EQ(xt::in1d(a, b.begin(), b.end()), res);
        EXPECT_EQ(xt::in1d(a, {1, 2}), res);
    }

    TEST(xset_operation, searchsorted)
    {
        xt::xtensor<size_t,1> a = {1, 2, 7, 8, 20};
        xt::xtensor<size_t,1> v = {9, 2, 2, 3, 22, 0};
        xt::xtensor<size_t,1> res_right = {4, 1, 1, 2, 5, 0};
        xt::xtensor<size_t,1> res_left = {4, 2, 2, 2, 5, 0};
        EXPECT_EQ(xt::searchsorted(a, v), res_right);
        EXPECT_EQ(xt::searchsorted(a, v, true), res_right);
        EXPECT_EQ(xt::searchsorted(a, v, false), res_left);
    }
}
