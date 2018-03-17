/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xfixed.hpp"

namespace xt
{
    using vector_type = svector<std::size_t, 4>;

    TEST(svector, behavior)
    {
        vector_type s = {1,2,3,4};
        vector_type s2 = s;
        std::vector<std::size_t> v(s.begin(), s.end());
        std::vector<std::size_t> v2 = {1,2,3,4};

        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));

        s.erase(s.begin(), s.begin() + 2);
        v.erase(v.begin(), v.begin() + 2);
        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));

        s2.erase(s2.begin() + 1, s2.end());
        v2.erase(v2.begin() + 1, v2.end());
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));

        EXPECT_TRUE(s2.on_stack());
        s2.push_back(10);
        s2.push_back(20);
        s2.push_back(30);
        s2.push_back(40);
        v2.push_back(10);
        v2.push_back(20);
        v2.push_back(30);
        v2.push_back(40);
        EXPECT_FALSE(s2.on_stack());
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));
    }

    TEST(svector, insert)
    {
        vector_type s = {1,2,3,4};
        vector_type s2 = s;
        std::vector<std::size_t> v(s.begin(), s.end());
        std::vector<std::size_t> v2 = {1,2,3,4};

        s.insert(s.begin(), std::size_t(55));
        s.insert(s.begin() + 2, std::size_t(123));
        v.insert(v.begin(), std::size_t(55));
        v.insert(v.begin() + 2, std::size_t(123));
        std::size_t nr = 12321;
        s.insert(s.end(), nr);
        v.insert(v.end(), nr);

        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));
    }

    TEST(svector, constructor)
    {
        vector_type a;
        EXPECT_EQ(size_t(0), a.size());

        vector_type b(10);
        EXPECT_EQ(size_t(10), b.size());

        vector_type c(10, 2);
        EXPECT_EQ(size_t(10), c.size());
        EXPECT_EQ(2, c[2]);

        std::vector<std::size_t> src(10, std::size_t(1));
        vector_type d(src.cbegin(), src.cend());
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(1, d[2]);

        vector_type e(src);
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(1, d[2]);
        
        vector_type f = { 1, 2, 3, 4 };
        EXPECT_EQ(size_t(4), f.size());
        EXPECT_EQ(3, f[2]);

        svector<std::size_t, 8> ov = { 1, 2, 3, 4, 5, 6, 7, 8 };
        vector_type g(ov);
        EXPECT_EQ(size_t(8), g.size());
        EXPECT_EQ(3, g[2]);
    }

    TEST(svector, assign)
    {
        vector_type a = { 1, 2, 3, 4 };
        
        vector_type src1(10, 2);
        a = src1;
        EXPECT_EQ(size_t(10), a.size());
        EXPECT_EQ(2, a[2]);

        std::vector<size_t> src2(5, 1);
        a = src2;
        EXPECT_EQ(size_t(5), a.size());
        EXPECT_EQ(1, a[2]);

        a = { 1, 2, 3, 4 };
        EXPECT_EQ(size_t(4), a.size());
        EXPECT_EQ(3, a[2]);

        svector<std::size_t, 4> src3(10, 1);
        a = src3;
        EXPECT_EQ(size_t(10), a.size());
        EXPECT_EQ(1, a[2]);
    }

    TEST(svector, resize)
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

    TEST(svector, access)
    {
        vector_type a(10);
        a[0] = 1;
        EXPECT_EQ(1, a[0]);
        a[3] = 3;
        EXPECT_EQ(3, a[3]);
        a[5] = 2;
        EXPECT_EQ(2, a[5]);

        a.front() = 0;
        EXPECT_EQ(0, a[0]);

        a.back() = 1;
        EXPECT_EQ(1, a[9]);
    }

    TEST(svector, iterator)
    {
        vector_type a(10);
        std::iota(a.begin(), a.end(), std::size_t(0));
        for (size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(i, a[i]);
        }
    }

    TEST(xshape, fixed)
    {
        fixed_shape<3, 4, 5> af;
        const_array<std::size_t, 3> a = af;
        EXPECT_EQ(a[0], 3);
        EXPECT_EQ(a[2], 5);
        EXPECT_EQ(a.back(), 5);
        EXPECT_EQ(a.front(), 3);
        EXPECT_EQ(a.size(), 3);
    }
}