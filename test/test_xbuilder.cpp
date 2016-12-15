/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    using std::size_t;

    TEST(xbuilder, ones)
    {
        std::vector<size_t> shape = {1, 2};
        auto lazy_ones = ones<double>(shape);
        xarray<double> assigned_ones = lazy_ones;
        ASSERT_EQ(2, lazy_ones.dimension());
        ASSERT_EQ(1, lazy_ones(0, 1));

        xarray<double> m1 {{ 1, 2, 3}, {4, 5, 6}};
        auto b = broadcast(m1, std::array<size_t, 3>{1, 2, 3});

        xarray<double> m2 {{ 1, 2, 3}, {4, 5, 6}};
        auto b2 = broadcast(m2, {1, 2, 3});
    }
}
