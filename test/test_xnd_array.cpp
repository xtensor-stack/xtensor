/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xnd_array.hpp"

namespace xt
{
    TEST(xnd_array, cast)
    {
        xarray<double> a1({{3.1, 4.1}, {5.1, 6.1}});

        xnd_array a2 = a1;
        a2.set_type("float64");

        auto a3 = a2.astype("int32");

        xarray<int> a4({{3, 4}, {5, 6}});
        auto a5 = a3.get<xarray<int32_t>>();

        EXPECT_EQ(a4, a5);
    }
}
