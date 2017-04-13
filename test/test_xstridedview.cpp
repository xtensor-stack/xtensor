/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstridedview.hpp"

#include "xtensor/xio.hpp"
#include <iostream>

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

    TEST(xstridedview, transpose_assignment)
    {
        xarray<double> e = xt::arange<double>(24);
        e.reshape({2, 2, 6});
        auto vt = transpose(e);

        vt(0, 0, 1) = 123;
        EXPECT_EQ(123, e(1, 0, 0));
        auto val = vt[{1,0,1}];
        EXPECT_EQ(e(1,0,1), val);
    }
}