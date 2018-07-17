/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xdynamic_view.hpp"
#include "xtensor/xtensor.hpp"

#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"

namespace xt
{
    using shape_t = std::vector<std::size_t>;
    using view_shape_type = dynamic_shape<std::size_t>;

    template <class T>
    struct DEBUG;

    TEST(xdynamic_view, keep)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));
        EXPECT_EQ(view0.dimension(), size_t(2));
        EXPECT_EQ(view0.shape()[0], size_t(2));
        EXPECT_EQ(view0.shape()[1], size_t(3));
        EXPECT_EQ(view0(0, 0), a(1, 0, 1));
        EXPECT_EQ(view0(0, 1), a(1, 0, 2));
        EXPECT_EQ(view0(0, 2), a(1, 0, 3));
        EXPECT_EQ(view0(1, 0), a(1, 2, 1));
        EXPECT_EQ(view0(1, 1), a(1, 2, 2));
        EXPECT_EQ(view0(1, 2), a(1, 2, 3));

        auto view1 = dynamic_view(a, xdynamic_slice_vector({ all(), 1, keep(0, 2, 3) }));
        EXPECT_EQ(view1.dimension(), size_t(2));
        EXPECT_EQ(view1.shape()[0], size_t(2));
        EXPECT_EQ(view1.shape()[1], size_t(3));
        EXPECT_EQ(view1(0, 0), a(0, 1, 0));
        EXPECT_EQ(view1(0, 1), a(0, 1, 2));
        EXPECT_EQ(view1(0, 2), a(0, 1, 3));
        EXPECT_EQ(view1(1, 0), a(1, 1, 0));
        EXPECT_EQ(view1(1, 1), a(1, 1, 2));
        EXPECT_EQ(view1(1, 2), a(1, 1, 3));

        auto view2 = dynamic_view(a, { all(), 1, keep(0, 2, 3) });
        EXPECT_EQ(view1, view2);

        auto view3 = dynamic_view(a, { all(), 1, newaxis(), keep(0, 2, 3) });
        EXPECT_EQ(view3.dimension(), size_t(3));
        EXPECT_EQ(view3.shape()[0], size_t(2));
        EXPECT_EQ(view3.shape()[1], size_t(1));
        EXPECT_EQ(view3.shape()[2], size_t(3));
        EXPECT_EQ(view3(0, 0, 0), a(0, 1, 0));
        EXPECT_EQ(view3(0, 0, 1), a(0, 1, 2));
        EXPECT_EQ(view3(0, 0, 2), a(0, 1, 3));
        EXPECT_EQ(view3(1, 0, 0), a(1, 1, 0));
        EXPECT_EQ(view3(1, 0, 1), a(1, 1, 2));
        EXPECT_EQ(view3(1, 0, 2), a(1, 1, 3));
    }

    TEST(xdynamic_view, keep_iterator)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));
        auto iter = view0.template begin<layout_type::row_major>();
        auto iter_end = view0.template end<layout_type::row_major>();

        EXPECT_EQ(*iter, a(1, 0, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 3));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 3));
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xdynamic_view, drop)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, drop(1), range(1, 4) }));
        EXPECT_EQ(view0.dimension(), size_t(2));
        EXPECT_EQ(view0.shape()[0], size_t(2));
        EXPECT_EQ(view0.shape()[1], size_t(3));
        EXPECT_EQ(view0(0, 0), a(1, 0, 1));
        EXPECT_EQ(view0(0, 1), a(1, 0, 2));
        EXPECT_EQ(view0(0, 2), a(1, 0, 3));
        EXPECT_EQ(view0(1, 0), a(1, 2, 1));
        EXPECT_EQ(view0(1, 1), a(1, 2, 2));
        EXPECT_EQ(view0(1, 2), a(1, 2, 3));

        auto view1 = dynamic_view(a, xdynamic_slice_vector({ all(), 1, drop(1, 2) }));
        EXPECT_EQ(view1.dimension(), size_t(2));
        EXPECT_EQ(view1.shape()[0], size_t(2));
        EXPECT_EQ(view1.shape()[1], size_t(2));
        EXPECT_EQ(view1(0, 0), a(0, 1, 0));
        EXPECT_EQ(view1(0, 1), a(0, 1, 3));
        EXPECT_EQ(view1(1, 0), a(1, 1, 0));
        EXPECT_EQ(view1(1, 1), a(1, 1, 3));

        auto view3 = dynamic_view(a, { all(), 1, newaxis(), drop(1, 2) });
        EXPECT_EQ(view3.dimension(), size_t(3));
        EXPECT_EQ(view3.shape()[0], size_t(2));
        EXPECT_EQ(view3.shape()[1], size_t(1));
        EXPECT_EQ(view3.shape()[2], size_t(2));
        EXPECT_EQ(view3(0, 0, 0), a(0, 1, 0));
        EXPECT_EQ(view3(0, 0, 1), a(0, 1, 3));
        EXPECT_EQ(view3(1, 0, 0), a(1, 1, 0));
        EXPECT_EQ(view3(1, 0, 1), a(1, 1, 3));
    }

    TEST(xdynamic_view, drop_iterator)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, drop(1), range(1, 4) }));
        auto iter = view0.template begin<layout_type::row_major>();
        auto iter_end = view0.template end<layout_type::row_major>();

        EXPECT_EQ(*iter, a(1, 0, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 3));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 3));
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xdynamic_view, semantic)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        xarray<int> b1 = { {13, 14, 15}, {21, 22, 23} };
        xarray<int> b2 = { {0, 1, 2}, {3, 4, 5} };
        xarray<int> exp = b1 + b2;
        auto view0 = dynamic_view(a, { 1, keep(0, 2), range(1, 4) });
        view0 += b2;
        EXPECT_EQ(view0, exp);
    }
}