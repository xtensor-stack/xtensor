/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xadapt.hpp"

namespace xt
{
    using vec_type = std::vector<int>;

    TEST(xarray_adaptor, xadapt)
    {
        vec_type v(4, 0);
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = xadapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[1]);

        shape_type str({2, 1});
        auto a2 = xadapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xarray_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = xadapt(data, size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[1]);

        shape_type str({2, 1});
        auto a2 = xadapt(data, size, no_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data[2]);

        delete[] data;
    }

    TEST(xarray_adaptor, pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        int* data2 = new int[size];
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = xadapt(data, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[1]);

        shape_type str({2, 1});
        auto a2 = xadapt(data2, size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xtensor_adaptor, xadapt)
    {
        vec_type v(4, 0);
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = xadapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[1]);

        shape_type str = {2, 1};
        auto a2 = xadapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xtensor_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = xadapt(data, size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[1]);

        shape_type str = {2, 1};
        auto a2 = xadapt(data, size, no_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data[2]);

        delete[] data;
    }

    TEST(xtensor_adaptor, pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        int* data2 = new int[size];
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = xadapt(data, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[1]);

        shape_type str = {2, 1};
        auto a2 = xadapt(data2, size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xtensor_adaptor, move_pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        int* data2 = new int[size];
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = xadapt(std::move(data), size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[1]);

        shape_type str = {2, 1};
        auto a2 = xadapt(std::move(data2), size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }
}
