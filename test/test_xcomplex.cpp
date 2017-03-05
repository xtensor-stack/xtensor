/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <complex>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp" 
#include "xtensor/xcomplex.hpp"

namespace xt
{
    using namespace std::complex_literals;

    TEST(xcomplex, real)
    {
        xarray<std::complex<double>> e =
            {{1.0       , 1.0 + 1.0i},
             {1.0 - 1.0i, 1.0       }};

        // Test real expression
        auto r = real(e);
        auto i = imag(e);

        ASSERT_EQ(r.dimension(), 2);
        ASSERT_EQ(i.dimension(), 2);

        ASSERT_EQ(r.shape()[0], 2);
        ASSERT_EQ(r.shape()[1], 2);
        ASSERT_EQ(i.shape()[0], 2);
        ASSERT_EQ(i.shape()[1], 2);
        
        ASSERT_EQ(i(0, 0), 0);
        ASSERT_EQ(i(0, 1), 1);
        ASSERT_EQ(i(1, 0), -1);
        ASSERT_EQ(i(1, 1), 0);

        // Test assignment to an array
        xarray<double> ar = r;
        EXPECT_TRUE(all(equal(ar, ones<double>({2, 2}))));

        // Test assigning an expression to the offset_view
        real(e) = zeros<double>({2, 2});
        xarray<std::complex<double>> expect1 = 
            {{0.0       , 0.0 + 1.0i},
             {0.0 - 1.0i, 0.0       }};
        EXPECT_TRUE(all(equal(e, expect1)));
        
        imag(e) = zeros<double>({2, 2});
        EXPECT_TRUE(all(equal(e, zeros<std::complex<double>>({2, 2}))));
    }
}

