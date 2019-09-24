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
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

namespace xt
{
    TEST(xexpression_traits, common_tensor_deduction)
    {
        xt::xarray<double> a1 {0.0, 0.0, 0.0};
        xt::xtensor<double, 1> a2 {0.0, 0.0, 0.0};
        xt::xtensor_fixed<double, xt::xshape<3>> a3({0.0, 0.0, 0.0});

        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a1), decltype(a1)>, decltype(a1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a2), decltype(a2)>, decltype(a2)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a3), decltype(a3)>, decltype(a3)>::value));

        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a1), decltype(a2)>, decltype(a1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a1), decltype(a3)>, decltype(a1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a2), decltype(a3)>, decltype(a2)>::value));

        auto sum1 = a1 + a2;
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a1), decltype(sum1)>, decltype(a1)>::value));
        auto sum2 = a1 + a3;
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a1), decltype(sum2)>, decltype(a1)>::value));
        auto sum3 = a2 + a3;
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(a2), decltype(sum3)>, decltype(a2)>::value));

        xt::xarray<double> b1 {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}};
        xt::xtensor<double, 2> b2 {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}};
        xt::xtensor_fixed<double, xt::xshape<3,3>> b3({{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}});

        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b1), decltype(b1)>, decltype(b1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b2), decltype(b2)>, decltype(b2)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b3), decltype(b3)>, decltype(b3)>::value));

        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b1), decltype(b2)>, decltype(b1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b1), decltype(b3)>, decltype(b1)>::value));
        EXPECT_TRUE((std::is_same<xt::common_tensor_type_t<decltype(b2), decltype(b3)>, decltype(b2)>::value));
    }
}

