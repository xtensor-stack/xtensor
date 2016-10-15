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

    TEST(operation, plus)
    {
        xshape<size_t> shape = {3 ,2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ((+a)(0, 0), +(a(0, 0)));
    }

    TEST(operation, minus)
    {
        xshape<size_t> shape = {3 ,2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ((-a)(0, 0), -(a(0, 0)));
    }

    TEST(operation, add)
    {
        xshape<size_t> shape = {3, 2};
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
        xshape<size_t> shape = {3, 2};
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
        xshape<size_t> shape = {3, 2};
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
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ((a / b)(0, 0), a(0, 0) / b(0, 0));
        
        double sb = 1.2;
        EXPECT_EQ((a / sb)(0, 0), a(0, 0) / sb);

        double sa = 4.6;
        EXPECT_EQ((sa / b)(0, 0), sa / b(0, 0));
    }
}

