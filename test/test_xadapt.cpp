/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xstrides.hpp"

namespace xt
{
    using vec_type = std::vector<int>;

    TEST(xarray_adaptor, adapt)
    {
        vec_type v(4, 0);
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[a1.strides()[1]]);

        shape_type str({2, 1});
        auto a2 = adapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xarray_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        int* data = new int[size];
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt(data, size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str({2, 1});
        auto a2 = adapt(data, size, no_ownership(), s, str);
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

        auto a1 = adapt(data, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str({2, 1});
        auto a2 = adapt(data2, size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xarray_adaptor, no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({ 1 });

        auto a1 = adapt(&data1, size, xt::no_ownership(), s);
        auto a2 = adapt(&data2, size, xt::no_ownership(), s);
        auto a3 = adapt(&data3, size, xt::no_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, data3);
    }

    TEST(xarray_adaptor, acquire_ownership_assign)
    {
        size_t size = 1;
        int* data1 = new int[1];
        data1[0] = 0;
        int* data2 = new int[1];
        data2[0] = 1;
        int* data3 = nullptr;
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({ 1 });

        auto a1 = adapt(data1, size, xt::acquire_ownership(), s);
        auto a2 = adapt(data2, size, xt::acquire_ownership(), s);
        auto a3 = adapt(data3, size_t(0), xt::acquire_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, *data3);
    }

    TEST(xtensor_adaptor, adapt)
    {
        vec_type v0(4, 0);
        auto a0 = adapt(v0);
        a0(0) = 1;
        a0(3) = 3;
        EXPECT_EQ(1, v0[0]);
        EXPECT_EQ(3, v0[3]);
        
        vec_type v(4, 0);
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xtensor_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        int* data = new int[size];

        auto a0 = adapt(data, size, no_ownership());
        a0(3) = 3;
        EXPECT_EQ(3, data[3]);

        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(data, size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(data, size, no_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data[2]);

        delete[] data;
    }

    TEST(xtensor_adaptor, pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data0 = new int[size];
        int* data1 = new int[size];
        int* data2 = new int[size];

        auto a0 = adapt(data0, size, acquire_ownership());
        a0(3) = 3;
        EXPECT_EQ(3, data0[3]);

        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(data1, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data1[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(data2, size, acquire_ownership(), s, str);
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

        auto a1 = adapt(std::move(data), size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(std::move(data2), size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xtensor_adaptor, no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(&data1, size, xt::no_ownership(), s);
        auto a2 = adapt(&data2, size, xt::no_ownership(), s);
        auto a3 = adapt(&data3, size, xt::no_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, data3);
    }

    TEST(xtensor_adaptor, const_no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        const int* const p_data1 = &data1;
        const int* const p_data2 = &data2;
        int* p_data3 = &data3;

        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(p_data1, size, xt::no_ownership(), s);
        auto a2 = adapt(p_data2, size, xt::no_ownership(), s);
        auto a3 = adapt(p_data3, size, xt::no_ownership(), s);

        a3 = a1 + a2;

        EXPECT_EQ(1, data3);
    }

    TEST(xtensor_adaptor, acquire_ownership_assign)
    {
        size_t size = 1;
        int* data1 = new int[1];
        data1[0] = 0;
        int* data2 = new int[1];
        data2[0] = 1;
        int* data3 = nullptr;
        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(data1, size, xt::acquire_ownership(), s);
        auto a2 = adapt(data2, size, xt::acquire_ownership(), s);
        auto a3 = adapt(data3, size_t(0), xt::acquire_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, *data3);
    }
}
