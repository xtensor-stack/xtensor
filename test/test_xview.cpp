/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include <algorithm>

namespace xt
{
    using std::size_t;
    using view_shape_type = std::vector<size_t>;

    TEST(xview, simple)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.storage_begin());

        auto view1 = make_xview(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(1, view1.dimension());

        auto view0 = make_xview(a, 0, range(0, 3));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(1, view0.dimension());
        EXPECT_EQ(3, view0.shape()[0]);

        auto view2 = make_xview(a, range(0, 2), 2);
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(1, view2.dimension());
        EXPECT_EQ(2, view2.shape()[0]);

        auto view4 = make_xview(a, 1);
        EXPECT_EQ(1, view4.dimension());
        EXPECT_EQ(4, view4.shape()[0]);

        auto view5 = make_xview(view4, 1);
        EXPECT_EQ(0, view5.dimension());
        EXPECT_EQ(0, view5.shape().size());

        auto view6 = make_xview(a, 1, all());
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = make_xview(a, all(), 2);
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));
    }

    TEST(xview, three_dimensional)
    {
        view_shape_type shape = {3, 4, 2};
        std::vector<double> data {
            1, 2,
            3, 4,
            5, 6,
            7, 8,

            9, 10,
            11, 12,
            21, 22, 
            23, 24,

            25, 26,
            27, 28,
            29, 210,
            211, 212
        };
        xarray<double> a(shape);
        std::copy(data.cbegin(), data.cend(), a.storage_begin());

        auto view1 = make_xview(a, 1);
        EXPECT_EQ(2, view1.dimension());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));
        
        std::array<std::size_t, 2> idx = {1, 1};
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xview, integral_count)
    {
        size_t squeeze1 = integral_count<size_t, size_t, size_t, xrange<size_t>>();
        EXPECT_EQ(squeeze1, 3);
        size_t squeeze2 = integral_count<size_t, xrange<size_t>, size_t>();
        EXPECT_EQ(squeeze2, 2);
        size_t squeeze3 = integral_count_before<size_t, size_t, size_t, xrange<size_t>>(3);
        EXPECT_EQ(squeeze3, 3);
        size_t squeeze4 = integral_count_before<size_t, xrange<size_t>, size_t>(2);
        EXPECT_EQ(squeeze4, 1);
    }

    TEST(xview, integral_skip)
    {
        size_t index0 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (0);
        size_t index1 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (1);
        size_t index2 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (2);
        EXPECT_EQ(index0, 1);
        EXPECT_EQ(index1, 3);
        EXPECT_EQ(index2, 4);
    }

    TEST(xview, iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double> a(shape);
        std::vector<double> data {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.storage_begin());

        auto view1 = make_xview(a, range(0, 2), 1, range(1, 4));
        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);

        auto view2 = make_xview(view1, range(0, 2), range(1, 3));
        auto iter2 = view2.begin();
        auto iter_end2 = view2.end();

        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
    }

    TEST(xview, xview_on_xfunction)
    {
        view_shape_type shape = {3, 4};
        xarray<int> a(shape);
        std::vector<int> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.storage_begin());

        view_shape_type shape2 = { 3 };
        xarray<int> b(shape2);
        std::vector<int> data2 = { 1, 2, 3 };
        std::copy(data2.cbegin(), data2.cend(), b.storage_begin());

        auto func = make_xview(a, 1, range(1, 4)) + b;
        auto iter = func.begin();
        auto iter_end = func.end();

        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(9, *iter);
        ++iter;
        EXPECT_EQ(11, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, xview_on_xtensor)
    {
        xtensor<int, 2> a({ 3, 4 });
        std::vector<int> data{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.storage_begin());

        auto view1 = make_xview(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(1, view1.dimension());

        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);

        xarray<int> b({ 3 }, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xview, trivial_iterating)
    {
        xtensor<double, 1> arr1{ {2} };
        std::fill(arr1.begin(), arr1.end(), 6);
        auto view = xt::make_xview(arr1, 0);
        auto iter = view.begin();
        auto iter_end = view.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, const_view)
    {
        const xtensor<double, 3> arr{ {1, 2, 3}, 2.5 };
        xtensor<double, 2> arr2{ {2, 3}, 0.0 };
        xtensor<double, 2> ref{ {2, 3}, 2.5 };
        arr2 = xt::make_xview(arr, 0);
        EXPECT_EQ(ref, arr2);
    }

    TEST(xview, newaxis_count)
    {
        size_t count1 = newaxis_count<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>();
        EXPECT_EQ(count1, 3);
        size_t count2 = newaxis_count<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>();
        EXPECT_EQ(count2, 2);
        size_t count3 = newaxis_count_before<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>(3);
        EXPECT_EQ(count3, 3);
        size_t count4 = newaxis_count_before<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>(2);
        EXPECT_EQ(count4, 1);
    }
}

