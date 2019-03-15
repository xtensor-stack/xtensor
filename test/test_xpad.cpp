/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xpad.hpp"

namespace xt
{
    TEST(xpad, constant)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 1, 2, 0, 0, 0},
                                    {0, 0, 0, 3, 4, 5, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, periodic)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5},
                                    {0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5},
                                    {0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::periodic);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0},
                                    {5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }
}
