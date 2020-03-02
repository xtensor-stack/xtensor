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
#include "xtensor/xaxis_slice_iterator.hpp"


namespace xt
{
    using std::size_t;

    xarray<int> get_slice_test_array()
    {
        xarray<int> res = { {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12}},
                           {{13, 14, 15, 16},
                            {17, 18, 19, 20},
                            {21, 22, 23, 24}} };
        return res;
    }

    TEST(xaxis_slice_iterator, begin)
    {
        xarray<int> a = get_slice_test_array();
        auto iter_begin = axis_slice_begin(a, 0);
        EXPECT_EQ(size_t(1), iter_begin->dimension());
        EXPECT_EQ(a.shape()[0], iter_begin->shape()[0]);
        EXPECT_EQ(a(0, 0, 0), (*iter_begin)(0));
        EXPECT_EQ(a(1, 0, 0), (*iter_begin)(1));
    }


    TEST(xaxis_slice_iterator, end)
    {
        xarray<int> a = get_slice_test_array();
        xarray<int, layout_type::column_major> a_col = get_slice_test_array();

        auto dist = std::distance(axis_slice_begin(a, 0), axis_slice_end(a, 0));
        EXPECT_EQ(12, dist);

        dist = std::distance(axis_slice_begin(a_col), axis_slice_end(a_col));
        EXPECT_EQ(12, dist);

        dist = std::distance(axis_slice_begin(a, 1), axis_slice_end(a, 1));
        EXPECT_EQ(8, dist);
       
        dist = std::distance(axis_slice_begin(a_col, 1), axis_slice_end(a_col, 1));
        EXPECT_EQ(8, dist);
       
        dist = std::distance(axis_slice_begin(a, 2), axis_slice_end(a, 2));
        EXPECT_EQ(6, dist);

        dist = std::distance(axis_slice_begin(a_col, 2), axis_slice_end(a_col, 2));
        EXPECT_EQ(6, dist);
    }

    TEST(xaxis_slice_iterator, increment)
    {
        xarray<int, layout_type::row_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, 0);
        ++iter;

        EXPECT_EQ(size_t(1), iter->dimension());
        EXPECT_EQ(a.shape()[0], iter->shape()[0]);

        EXPECT_EQ(a(0, 0, 1), (*iter)(0));
        EXPECT_EQ(a(1, 0, 1), (*iter)(1));
    }

    TEST(xaxis_slice_iterator, const_array)
    {
        const xarray<int, layout_type::row_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, 2);
        ++iter;

        EXPECT_EQ(size_t(1), iter->dimension());
        EXPECT_EQ(a.shape()[2], iter->shape()[0]);

        EXPECT_EQ(a(0, 1, 0), (*iter)(0));
        EXPECT_EQ(a(0, 1, 1), (*iter)(1));
        EXPECT_EQ(a(0, 1, 2), (*iter)(2));
        EXPECT_EQ(a(0, 1, 3), (*iter)(3));
    }


    TEST(xaxis_slice_iterator, axis_0)
    {
        xarray<int, layout_type::row_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(0));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 0, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 1), (*iter)(0));
        EXPECT_EQ(a(1, 0, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 2), (*iter)(0));
        EXPECT_EQ(a(1, 0, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 3), (*iter)(0));
        EXPECT_EQ(a(1, 0, 3), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 1), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 2), (*iter)(0));
        EXPECT_EQ(a(1, 1, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 3), (*iter)(0));
        EXPECT_EQ(a(1, 1, 3), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 0), (*iter)(0));
        EXPECT_EQ(a(1, 2, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 1), (*iter)(0));
        EXPECT_EQ(a(1, 2, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 2), (*iter)(0));
        EXPECT_EQ(a(1, 2, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 3), (*iter)(0));
        EXPECT_EQ(a(1, 2, 3), (*iter)(1));
    }

    TEST(xaxis_slice_iterator, axis_0_col)
    {
        xarray<int, layout_type::column_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(0));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 0, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 0), (*iter)(0));
        EXPECT_EQ(a(1, 2, 0), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 1), (*iter)(0));
        EXPECT_EQ(a(1, 0, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 1), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 1), (*iter)(0));
        EXPECT_EQ(a(1, 2, 1), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 2), (*iter)(0));
        EXPECT_EQ(a(1, 0, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 2), (*iter)(0));
        EXPECT_EQ(a(1, 1, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 2), (*iter)(0));
        EXPECT_EQ(a(1, 2, 2), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 0, 3), (*iter)(0));
        EXPECT_EQ(a(1, 0, 3), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 1, 3), (*iter)(0));
        EXPECT_EQ(a(1, 1, 3), (*iter)(1));
        ++iter;
        EXPECT_EQ(a(0, 2, 3), (*iter)(0));
        EXPECT_EQ(a(1, 2, 3), (*iter)(1));
    }

    TEST(xaxis_slice_iterator, axis_1)
    {
        xarray<int, layout_type::row_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(1));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(0, 1, 0), (*iter)(1));
        EXPECT_EQ(a(0, 2, 0), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 1), (*iter)(0));
        EXPECT_EQ(a(0, 1, 1), (*iter)(1));
        EXPECT_EQ(a(0, 2, 1), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 2), (*iter)(0));
        EXPECT_EQ(a(0, 1, 2), (*iter)(1));
        EXPECT_EQ(a(0, 2, 2), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 3), (*iter)(0));
        EXPECT_EQ(a(0, 1, 3), (*iter)(1));
        EXPECT_EQ(a(0, 2, 3), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 0), (*iter)(1));
        EXPECT_EQ(a(1, 2, 0), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 1), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        EXPECT_EQ(a(1, 2, 1), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 2), (*iter)(0));
        EXPECT_EQ(a(1, 1, 2), (*iter)(1));
        EXPECT_EQ(a(1, 2, 2), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 3), (*iter)(0));
        EXPECT_EQ(a(1, 1, 3), (*iter)(1));
        EXPECT_EQ(a(1, 2, 3), (*iter)(2));
    }

    TEST(xaxis_slice_iterator, axis_1_col)
    {
        xarray<int, layout_type::column_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(1));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(0, 1, 0), (*iter)(1));
        EXPECT_EQ(a(0, 2, 0), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 0), (*iter)(1));
        EXPECT_EQ(a(1, 2, 0), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 1), (*iter)(0));
        EXPECT_EQ(a(0, 1, 1), (*iter)(1));
        EXPECT_EQ(a(0, 2, 1), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 1), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        EXPECT_EQ(a(1, 2, 1), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 2), (*iter)(0));
        EXPECT_EQ(a(0, 1, 2), (*iter)(1));
        EXPECT_EQ(a(0, 2, 2), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 2), (*iter)(0));
        EXPECT_EQ(a(1, 1, 2), (*iter)(1));
        EXPECT_EQ(a(1, 2, 2), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(0, 0, 3), (*iter)(0));
        EXPECT_EQ(a(0, 1, 3), (*iter)(1));
        EXPECT_EQ(a(0, 2, 3), (*iter)(2));
        ++iter;
        EXPECT_EQ(a(1, 0, 3), (*iter)(0));
        EXPECT_EQ(a(1, 1, 3), (*iter)(1));
        EXPECT_EQ(a(1, 2, 3), (*iter)(2));
    }


    TEST(xaxis_slice_iterator, axis_2)
    {
        xarray<int, layout_type::row_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(2));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(0, 0, 1), (*iter)(1));
        EXPECT_EQ(a(0, 0, 2), (*iter)(2));
        EXPECT_EQ(a(0, 0, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(0, 1, 0), (*iter)(0));
        EXPECT_EQ(a(0, 1, 1), (*iter)(1));
        EXPECT_EQ(a(0, 1, 2), (*iter)(2));
        EXPECT_EQ(a(0, 1, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(0, 2, 0), (*iter)(0));
        EXPECT_EQ(a(0, 2, 1), (*iter)(1));
        EXPECT_EQ(a(0, 2, 2), (*iter)(2));
        EXPECT_EQ(a(0, 2, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 0, 1), (*iter)(1));
        EXPECT_EQ(a(1, 0, 2), (*iter)(2));
        EXPECT_EQ(a(1, 0, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 1, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        EXPECT_EQ(a(1, 1, 2), (*iter)(2));
        EXPECT_EQ(a(1, 1, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 2, 0), (*iter)(0));
        EXPECT_EQ(a(1, 2, 1), (*iter)(1));
        EXPECT_EQ(a(1, 2, 2), (*iter)(2));
        EXPECT_EQ(a(1, 2, 3), (*iter)(3));
    }

    TEST(xaxis_slice_iterator, axis_2_col)
    {
        xarray<int, layout_type::column_major> a = get_slice_test_array();
        auto iter = axis_slice_begin(a, size_t(2));

        EXPECT_EQ(a(0, 0, 0), (*iter)(0));
        EXPECT_EQ(a(0, 0, 1), (*iter)(1));
        EXPECT_EQ(a(0, 0, 2), (*iter)(2));
        EXPECT_EQ(a(0, 0, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 0, 0), (*iter)(0));
        EXPECT_EQ(a(1, 0, 1), (*iter)(1));
        EXPECT_EQ(a(1, 0, 2), (*iter)(2));
        EXPECT_EQ(a(1, 0, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(0, 1, 0), (*iter)(0));
        EXPECT_EQ(a(0, 1, 1), (*iter)(1));
        EXPECT_EQ(a(0, 1, 2), (*iter)(2));
        EXPECT_EQ(a(0, 1, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 1, 0), (*iter)(0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1));
        EXPECT_EQ(a(1, 1, 2), (*iter)(2));
        EXPECT_EQ(a(1, 1, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(0, 2, 0), (*iter)(0));
        EXPECT_EQ(a(0, 2, 1), (*iter)(1));
        EXPECT_EQ(a(0, 2, 2), (*iter)(2));
        EXPECT_EQ(a(0, 2, 3), (*iter)(3));
        ++iter;
        EXPECT_EQ(a(1, 2, 0), (*iter)(0));
        EXPECT_EQ(a(1, 2, 1), (*iter)(1));
        EXPECT_EQ(a(1, 2, 2), (*iter)(2));
        EXPECT_EQ(a(1, 2, 3), (*iter)(3));
    }
}
