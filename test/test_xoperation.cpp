/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
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

    TEST(operation, less)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {1, 1, 1, 0, 0};
        xarray<bool> b = a < 4; 
        EXPECT_EQ(expected, b);
    }

    TEST(operation, less_equal)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {1, 1, 1, 1, 0};
        xarray<bool> b = a <= 4; 
        EXPECT_EQ(expected, b);
    }

    TEST(operation, greater)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {0, 0, 0, 0, 1};
        xarray<bool> b = a > 4;
        EXPECT_EQ(expected, b);
    }

    TEST(operation, greater_equal)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {0, 0, 0, 1, 1};
        xarray<bool> b = a >= 4; 
        EXPECT_EQ(expected, b);
    }

    TEST(operation, negate)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {1, 1, 1, 0, 0};
        xarray<bool> b = !(a >= 4);
        EXPECT_EQ(expected, b);
    }

    TEST(operation, equal)
    {
        xarray<double> a = {1, 2, 3, 4, 5};
        xarray<bool> expected = {0, 0, 0, 1, 0};
        xarray<bool> b = equal(a, 4);
        EXPECT_EQ(expected, b);

        xarray<double> other = {1, 2, 3, 0, 0};
        xarray<bool> b_2 = equal(a, other);
        xarray<bool> expected_2 = {1, 1, 1, 0, 0};
        EXPECT_EQ(expected_2, b_2);
    }

    TEST(operation, not_equal)
    {
        xarray<double> a = { 1, 2, 3, 4, 5 };
        xarray<bool> expected = { 1, 1, 1, 0, 1 };
        xarray<bool> b = not_equal(a, 4);
        EXPECT_EQ(expected, b);

        xarray<double> other = { 1, 2, 3, 0, 0 };
        xarray<bool> b_2 = not_equal(a, other);
        xarray<bool> expected_2 = { 0, 0, 0, 1, 1 };
        EXPECT_EQ(expected_2, b_2);
    }

    TEST(operation, logical_and)
    {
        xarray<bool> a = {0, 0, 0, 1, 0};
        xarray<bool> expected = {0, 0, 0, 0, 0};
        xarray<bool> b = a && 0;
        xarray<bool> c = a && a;
        EXPECT_EQ(expected, b);
        EXPECT_EQ(c, a);
    }

    TEST(operation, logical_or)
    {
        xarray<bool> a = {0, 0, 0, 1, 0};
        xarray<bool> other = {0, 0, 0, 0, 0};
        xarray<bool> b = a || other;
        xarray<bool> c = a || 0;
        xarray<bool> d = a || 1;
        EXPECT_EQ(b, a);
        EXPECT_EQ(c, a);

        xarray<bool>expected = {1, 1, 1, 1, 1};
        EXPECT_EQ(expected, d);
    }

    TEST(operation, any)
    {
        xarray<int> a = {0, 0, 3};
        EXPECT_EQ(true, any(a));
        xarray<int> b = {{0, 0, 0}, {0, 0, 0}};
        EXPECT_EQ(false, any(b));
    }

    TEST(operation, minimum)
    {
        xarray<int> a = {0, 0, 3};
        xarray<int> b = {-1, 0, 10};
        xarray<int> expected = {-1, 0, 3};
        EXPECT_TRUE(all(equal(minimum(a, b), expected)));
    }

    TEST(operation, maximum)
    {
        xarray<int> a = {0, 0, 3};
        xarray<int> b = {-1, 0, 10};
        xarray<int> expected = {0, 0, 10};
        xarray<int> expected_2 = {0, 1, 10};
        EXPECT_TRUE(all(equal(maximum(a, b), expected)));
        EXPECT_TRUE(all(equal(maximum(arange(0, 3), b), expected_2)));
    }

    TEST(operation, amax)
    {
        xarray<int> a = {{0, 0, 3}, {1,2, 10}};
        EXPECT_EQ(10, amax(a)());
        xarray<int> e1 = {1, 2, 10};
        EXPECT_EQ(e1, amax(a, {0}));
        xarray<int> e2 = {3, 10};
        EXPECT_EQ(e2, amax(a, {1}));
    }

    TEST(operation, amin)
    {
        xarray<int> a = {{0, 0, 3}, {1,2, 10}};
        EXPECT_EQ(0, amin(a)());
        xarray<int> e1 = {0, 0, 3};
        EXPECT_EQ(e1, amin(a, {0}));
        xarray<int> e2 = {0, 1};
        EXPECT_EQ(e2, amin(a, {1}));
    }

    TEST(operation, all)
    {
        xarray<int> a = {1, 1, 3};
        EXPECT_EQ(true, all(a));
        xarray<int> b = {{0, 2, 1}, {2, 1, 0}};
        EXPECT_EQ(false, all(b));
    }

    TEST(operation, all_layout)
    {
        xarray<int, layout_type::row_major> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<int, layout_type::column_major> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        EXPECT_EQ(a(0, 1), b(0, 1));
        EXPECT_TRUE(all(equal(a, b)));
    }

    TEST(operation, nonzero)
    {
        xarray<int> a = {1, 0, 3};
        std::vector<xindex> expected = {{0}, {2}};
        EXPECT_EQ(expected, nonzero(a));

        xarray<int> b = {{0, 2, 1}, {2, 1, 0}};
        std::vector<xindex> expected_b = {{0, 1}, {0, 2}, {1, 0}, {1, 1}};
        EXPECT_EQ(expected_b, nonzero(b));

        auto c = equal(b, 0);
        std::vector<xindex> expected_c = {{0, 0}, {1, 2}};
        EXPECT_EQ(expected_c, nonzero(c));

        shape_type s = {3, 3, 3};
        xarray<bool> d(s);
        std::fill(d.xbegin(), d.xend(), true);

        auto d_nz = nonzero(d);
        EXPECT_EQ(3 * 3 * 3, d_nz.size());
        xindex last_idx = {2, 2, 2};
        EXPECT_EQ(last_idx, d_nz.back());
    }

    TEST(operation, where_only_condition)
    {
        xarray<int> a = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        std::vector<xindex> expected = {{0, 0}, {1, 1}, {2, 2}};
        EXPECT_EQ(expected, where(a));
    }
}

