/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xarray/xscalar.hpp"
#include "xarray/xindex.hpp"

namespace xt
{
    TEST(xscalar, size)
    {
        // The shape of a 0-D xarray is ().  The size of the buffer is 1.
        xscalar<int> x(1);
        ASSERT_TRUE(x.size() == data_size(x.shape()));
    }

    TEST(xscalar, access)
    {
        // Calling operator() with no argument returns the wrapped value.
        xscalar<int> x(2);
        ASSERT_TRUE(x() == 2);
    }

    TEST(xscalar, dimension)
    {
        // The dimension of a xscalar is 0
        xscalar<int> x(2);
        ASSERT_TRUE(x.dimension() == 0);
    }
}

