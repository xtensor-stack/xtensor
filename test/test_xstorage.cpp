/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xtensor_config.hpp"
#include "xtensor/xstorage.hpp"
#include <numeric>

namespace xt
{
    using vector_type = uvector<double, DEFAULT_ALLOCATOR(double)>;

    TEST(uvector, constructor)
    {
        vector_type a;
        EXPECT_EQ(size_t(0), a.size());

        vector_type b(10);
        EXPECT_EQ(size_t(10), b.size());

        vector_type c(10, 2.5);
        EXPECT_EQ(size_t(10), c.size());
        EXPECT_EQ(2.5, c[2]);

        std::vector<double> src(10, 1.5);
        vector_type d(src.cbegin(), src.cend());
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(1.5, d[2]);
    }

    TEST(uvector, resize)
    {
        vector_type a;
        for (size_t i = 1; i < 11; ++i)
        {
            size_t size1 = i * 10;
            a.resize(size1);
            EXPECT_EQ(size1, a.size());
            size_t size2 = size1 - 5;
            a.resize(size2);
            EXPECT_EQ(size2, a.size());
        }
    }

    TEST(uvector, access)
    {
        vector_type a(10);
        a[0] = 1.0;
        EXPECT_EQ(1.0, a[0]);
        a[3] = 3.2;
        EXPECT_EQ(3.2, a[3]);
        a[5] = 2.7;
        EXPECT_EQ(2.7, a[5]);

        a.front() = 0.0;
        EXPECT_EQ(0.0, a[0]);

        a.back() = 1.0;
        EXPECT_EQ(1.0, a[9]);
    }

    TEST(uvector, iterator)
    {
        vector_type a(10);
        std::iota(a.begin(), a.end(), 0.);
        for (size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(double(i), a[i]);
        }
    }
}
