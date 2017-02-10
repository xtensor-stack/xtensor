/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xscalar.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xscalar, size)
    {
        // The shape of a 0-D xarray is ().  The size of the buffer is 1.
        xscalar<int> x(1);
        EXPECT_EQ(x.size(), 1);
    }

    TEST(xscalar, access)
    {
        // Calling operator() with no argument returns the wrapped value.
        xscalar<int> x(2);
        EXPECT_EQ(x(), 2);

        x() = 4;
        EXPECT_EQ(4, x());
    }

    TEST(xscalar, dimension)
    {
        // The dimension of a xscalar is 0
        xscalar<int> x(2);
        EXPECT_EQ(x.dimension(), 0);
    }

    TEST(xscalar, iterator)
    {
        xscalar<int> x(2);
        auto iter = x.xbegin();
        *iter = 4;
        EXPECT_EQ(4, x());
    }

    TEST(xscalar, xref)
    {
        int ref = 4;
        int x = 2;
        auto s = xref(x);
        s() = ref;
        EXPECT_EQ(ref, x);
    }
}

