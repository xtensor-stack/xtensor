/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <cstddef>
#include "xtensor/xarray.hpp"

namespace xt
{
    using std::size_t;
    using shape_type = std::vector<size_t>;

    TEST(operation, plus)
    {
        shape_type shape = {3 ,2};
        xarray<double> a(shape, 4.5);
		double ref = +(a(0, 0));
		double actual = (+a)(0, 0);
        EXPECT_EQ(ref, actual);
    }

    TEST(operation, minus)
    {
        shape_type shape = {3 ,2};
        xarray<double> a(shape, 4.5);
		double ref = -(a(0, 0));
		double actual = (-a)(0, 0);
        EXPECT_EQ(ref, actual);
    }

    TEST(operation, add)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ((a + b)(0, 0), a(0, 0) + b(0, 0));
        
        double sb = 1.2;
        EXPECT_EQ((a + sb)(0, 0), a(0, 0) + sb);

        double sa = 4.6;
        EXPECT_EQ((sa + b)(0, 0), sa + b(0, 0));
    }

    TEST(operation, subtract)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ((a - b)(0, 0), a(0, 0) - b(0, 0));
        
        double sb = 1.2;
        EXPECT_EQ((a - sb)(0, 0), a(0, 0) - sb);

        double sa = 4.6;
        EXPECT_EQ((sa - b)(0, 0), sa - b(0, 0));
    }
    
    TEST(operation, multiply)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ((a * b)(0, 0), a(0, 0) * b(0, 0));
        
        double sb = 1.2;
        EXPECT_EQ((a * sb)(0, 0), a(0, 0) * sb);

        double sa = 4.6;
        EXPECT_EQ((sa * b)(0, 0), sa * b(0, 0));
    }
    
    TEST(operation, divide)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ((a / b)(0, 0), a(0, 0) / b(0, 0));
        
        double sb = 1.2;
        EXPECT_EQ((a / sb)(0, 0), a(0, 0) / sb);

        double sa = 4.6;
        EXPECT_EQ((sa / b)(0, 0), sa / b(0, 0));
    }
}

