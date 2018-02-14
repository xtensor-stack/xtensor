/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    using xtensorf3x3 = xtensorf<double, xt::xshape<3, 3>>; 
    using xtensorf3 = xtensorf<double, xt::xshape<3>>; 

    TEST(xtensorf, basic)
    {
        xtensorf3x3 a({{1,2,3}, {4,5,6}, {7,8,9}});
        xtensorf3x3 b({{1,2,3}, {4,5,6}, {7,8,9}});

        xtensorf3x3 res1 = a + b;
        xtensorf3x3 res2 = 2.0 * a;

        EXPECT_EQ(res1, res2);
    }

    TEST(xtensorf, broadcast)
    {
        xtensorf3x3 a({{1,2,3}, {4,5,6}, {7,8,9}});
        xtensorf3 b({4,5,6});

        xtensorf3x3 res = a * b;
        xtensorf3x3 resb = b * a;

        xarray<double> ax = a;
        xarray<double> bx = b;
        xarray<double> arx = a * b;
        xarray<double> brx = b * a;

        EXPECT_EQ(res, arx);
        EXPECT_EQ(resb, brx);

#ifdef XTENSOR_ENABLE_ASSERT
        EXPECT_THROW(a.resize({2, 2}), std::runtime_error);
#endif
        // reshaping fixed container
        EXPECT_THROW(a.reshape({1, 9}), std::runtime_error);
        EXPECT_NO_THROW(a.reshape({3, 3}));
        EXPECT_NO_THROW(a.reshape({3, 3}, DEFAULT_LAYOUT));
        EXPECT_THROW(a.reshape({3, 3}, layout_type::any), std::runtime_error);
    }

    TEST(xtensorf, adapt)
    {
        std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        xfixed_adaptor<decltype(a), xt::xshape<3, 3>> ad(a);
        xtensorf3x3 b({{1,2,3}, {4,5,6}, {7,8,9}});

        EXPECT_EQ(5.0, ad(1, 1));
        auto expr = ad + b;
        EXPECT_EQ(8, expr(1, 0));
        ad = b * 2;
        EXPECT_EQ(ad(0, 1), 2 * 2);
    }
}