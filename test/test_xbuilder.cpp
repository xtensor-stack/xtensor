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
    }
}
