/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include <algorithm>

namespace xt
{
    using std::size_t;

    TEST(xview, simple)
    {
        xshape<size_t> shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.begin(), data.end(), a.storage_begin());

        auto view1 = make_xview(a, 1, range(1, 4));
        ASSERT_EQ(view1(0), a(1, 1));
        ASSERT_EQ(view1(1), a(1, 2));
        ASSERT_EQ(view1.dimension(), 1);

        auto view2 = make_xview(a, 0, range(0, 3));
        ASSERT_EQ(view2(0), a(0, 0));
        ASSERT_EQ(view2(1), a(0, 1));
        ASSERT_EQ(view2.dimension(), 1);
        ASSERT_EQ(view2.shape()[0], 3);

        auto view3 = make_xview(a, range(0, 2), 2);
        ASSERT_EQ(view3(0), a(0, 2));
        ASSERT_EQ(view3(1), a(1, 2));
        ASSERT_EQ(view3.dimension(), 1);
        ASSERT_EQ(view3.shape()[0], 2);

        auto view4 = make_xview(a, 1);
        ASSERT_EQ(view4.dimension(), 1);
        ASSERT_EQ(view4.shape()[0], 4);

        auto view5 = make_xview(view4, 1);
        ASSERT_EQ(view5.dimension(), 0);
        ASSERT_EQ(view5.shape().size(), 0);
    }

    TEST(xview, integral_count)
    {
        size_t squeeze1 = integral_count<size_t, size_t, size_t, xrange<size_t>>();
        ASSERT_EQ(squeeze1, 3);
        size_t squeeze2 = integral_count<size_t, xrange<size_t>, size_t>();
        ASSERT_EQ(squeeze2, 2);
        size_t squeeze3 = integral_count_before<size_t, size_t, size_t, xrange<size_t>>(3);
        ASSERT_EQ(squeeze3, 3);
        size_t squeeze4 = integral_count_before<size_t, xrange<size_t>, size_t>(2);
        ASSERT_EQ(squeeze4, 1);
    }

    TEST(xview, integral_skip)
    {
        size_t index0 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (0);
        size_t index1 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (1);
        size_t index2 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>> (2);
        ASSERT_EQ(index0, 1);
        ASSERT_EQ(index1, 3);
        ASSERT_EQ(index2, 4);
    }
}

