/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include <algorithm>

namespace xt
{
    using std::size_t;
    using view_shape_type = dynamic_shape<size_t>;

    TEST(xdynamic_view, simple)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = dynamic_view(a, slice_vector({1, range(1, 4)}));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());

        auto view0 = dynamic_view(a, slice_vector({0, range(0, 3)}));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(size_t(1), view0.dimension());
        EXPECT_EQ(size_t(3), view0.shape()[0]);

        auto view2 = dynamic_view(a, slice_vector({range(0, 2), 2}));
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(2), view2.shape()[0]);

        auto view4 = dynamic_view(a, {1});
        EXPECT_EQ(size_t(1), view4.dimension());
        EXPECT_EQ(size_t(4), view4.shape()[0]);

        auto view5 = dynamic_view(view4, {1});
        EXPECT_EQ(size_t(0), view5.dimension());
        EXPECT_EQ(size_t(0), view5.shape().size());

        auto view6 = dynamic_view(a, slice_vector({1, all()}));
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = dynamic_view(a, slice_vector({all(), 2}));
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));

        EXPECT_THROW(dynamic_view(a, {all(), all(), 1}), std::runtime_error);
        EXPECT_THROW(dynamic_view(a, {all(), all(), all()}), std::runtime_error);
        EXPECT_NO_THROW(dynamic_view(a, {all(), newaxis(), all()}));
        EXPECT_NO_THROW(dynamic_view(a, {3, newaxis(), 1}));
        EXPECT_NO_THROW(dynamic_view(a, {3, 1, newaxis()}));
    }

    TEST(xdynamic_view, three_dimensional)
    {
        view_shape_type shape = {3, 4, 2};
        std::vector<double> data = {
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
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = dynamic_view(a, {1});
        EXPECT_EQ(size_t(2), view1.dimension());
        view_shape_type expected_shape = {4, 2};
        EXPECT_EQ(expected_shape, view1.shape());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));

        std::array<std::size_t, 2> idx = {1, 1};
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xdynamic_view, iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double> a(shape);
        std::vector<double> data {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = dynamic_view(a, slice_vector({range(0, 2), 1, range(1, 4)}));
        auto iter = view1.template begin<layout_type::row_major>();
        auto iter_end = view1.template end<layout_type::row_major>();

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
        EXPECT_FALSE(iter < iter_end);

        auto view2 = dynamic_view(view1, slice_vector({range(0, 2), range(1, 3)}));
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter_end2 = view2.template end<layout_type::row_major>();

        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
        EXPECT_FALSE(iter2 < iter_end2);
    }

    TEST(xdynamic_view, xdynamic_view_on_xfunction)
    {
        xarray<int> a = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12} };
        xarray<int> b = { 1, 2, 3, 4 };

        auto sum = a + b;
        auto func = dynamic_view(sum, slice_vector({1, range(1, 4)}));
        auto iter = func.template begin<layout_type::row_major>();
        auto iter_end = func.template end<layout_type::row_major>();

        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(10, *iter);
        ++iter;
        EXPECT_EQ(12, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(dynamic_view, view_on_generator)
    {
        auto vgen = dynamic_view(eye(4), slice_vector({1, range(1, 4)}));
        auto iter = vgen.cbegin();
    }

    TEST(xdynamic_view, xdynamic_view_on_xtensor)
    {
        xtensor<int, 2> a({3, 4});
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = dynamic_view(a, slice_vector({1, range(1, 4)}));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());

        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);

        xarray<int> b({3}, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xdynamic_view, const_view)
    {
        const xtensor<double, 3> arr{{1, 2, 3}, 2.5};
        xtensor<double, 2> arr2{{2, 3}, 0.0};
        xtensor<double, 2> ref{{2, 3}, 2.5};
        arr2 = dynamic_view(arr, {0});
        EXPECT_EQ(ref, arr2);
        // check that the following compiles
        auto v = dynamic_view(arr, {0});
        double acc = v(0);
        EXPECT_EQ(acc, 2.5);
        auto iter = v.begin();
    }

    TEST(xdynamic_view, newaxis)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = dynamic_view(a, slice_vector({all(), newaxis(), all()}));
        EXPECT_EQ(a(1, 1), view1(1, 0, 1));
        EXPECT_EQ(a(1, 2), view1(1, 0, 2));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape()[0]);
        EXPECT_EQ(size_t(1), view1.shape()[1]);
        EXPECT_EQ(size_t(4), view1.shape()[2]);

        auto view2 = dynamic_view(a, slice_vector({all(), all(), newaxis()}));
        EXPECT_EQ(a(1, 1), view2(1, 1, 0));
        EXPECT_EQ(a(1, 2), view2(1, 2, 0));
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape()[0]);
        EXPECT_EQ(size_t(4), view2.shape()[1]);
        EXPECT_EQ(size_t(1), view2.shape()[2]);

        auto view3 = dynamic_view(a, slice_vector({1, newaxis(), all()}));
        EXPECT_EQ(a(1, 1), view3(0, 1));
        EXPECT_EQ(a(1, 2), view3(0, 2));
        EXPECT_EQ(size_t(2), view3.dimension());

        auto view4 = dynamic_view(a, slice_vector({1, all(), newaxis()}));
        EXPECT_EQ(a(1, 1), view4(1, 0));
        EXPECT_EQ(a(1, 2), view4(2, 0));
        EXPECT_EQ(size_t(2), view4.dimension());

        auto view5 = dynamic_view(view1, {1});
        EXPECT_EQ(a(1, 1), view5(0, 1));
        EXPECT_EQ(a(1, 2), view5(0, 2));
        EXPECT_EQ(size_t(2), view5.dimension());

        auto view6 = dynamic_view(view2, {1});
        EXPECT_EQ(a(1, 1), view6(1, 0));
        EXPECT_EQ(a(1, 2), view6(2, 0));
        EXPECT_EQ(size_t(2), view6.dimension());

        std::array<std::size_t, 3> idx1 = {1, 0, 2};
        EXPECT_EQ(a(1, 2), view1.element(idx1.begin(), idx1.end()));

        std::array<std::size_t, 3> idx2 = {1, 2, 0};
        EXPECT_EQ(a(1, 2), view2.element(idx2.begin(), idx2.end()));
    }

    TEST(xdynamic_view, newaxis_iterating)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = dynamic_view(a, slice_vector({all(), all(), newaxis()}));
        auto iter1 = view1.template begin<layout_type::row_major>();
        auto iter1_end = view1.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 3), *iter1);
        ++iter1;
        EXPECT_EQ(iter1_end, iter1);

        auto view2 = dynamic_view(a, slice_vector({all(), newaxis(), all()}));
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter2_end = view2.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 3), *iter2);
        ++iter2;
        EXPECT_EQ(iter2_end, iter2);
    }

    TEST(xdynamic_view, newaxis_function)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        xarray<double> b(view_shape_type(1, 4));
        auto data_end = data.cbegin();
        data_end += 4;
        std::copy(data.cbegin(), data_end, b.template begin<layout_type::row_major>());

        auto v = dynamic_view(b, slice_vector({newaxis(), all()}));
        xarray<double> res = a + v;

        std::vector<double> data2{2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16};
        xarray<double> expected(shape);
        std::copy(data2.cbegin(), data2.cend(), expected.template begin<layout_type::row_major>());

        EXPECT_EQ(expected, res);
    }

    TEST(xdynamic_view, range_adaptor)
    {
        using namespace xt::placeholders;
        using t = xarray<int>;
        t a = {1, 2, 3, 4, 5};

        auto n = xnone();

        auto v1 = dynamic_view(a, slice_vector({range(3, _)}));
        t v1e = {4, 5};
        EXPECT_TRUE(v1e == v1);

        auto v2 = dynamic_view(a, slice_vector({range(_, 2)}));
        t v2e = {1, 2};
        EXPECT_TRUE(v2e == v2);

        auto v3 = dynamic_view(a, slice_vector({range(n, n)}));
        t v3e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v3e == v3);

        auto v4 = dynamic_view(a, slice_vector({range(n, 2, -1)}));
        t v4e = {5, 4};
        EXPECT_TRUE(v4e == v4);

        auto v5 = dynamic_view(a, slice_vector({range(2, n, -1)}));
        t v5e = {3, 2, 1};
        EXPECT_TRUE(v5e == v5);

        auto v6 = dynamic_view(a, slice_vector({range(n, n, n)}));
        t v6e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v6e == v6);

        auto v7 = dynamic_view(a, slice_vector({range(1, n, 2)}));
        t v7e = {2, 4};
        EXPECT_TRUE(v7e == v7);

        auto v8 = dynamic_view(a, slice_vector({range(2, n, 2)}));
        t v8e = {3, 5};
        EXPECT_TRUE(v8e == v8);
    }

    TEST(xdynamic_view, assign)
    {
        using t = xarray<int>;
        t a = {1, 2, 3, 4, 5};

        auto v = dynamic_view(a, slice_vector({range(0, 2)}));
        v = 1000;
        EXPECT_EQ(v(0), 1000);
        EXPECT_EQ(a(0), 1000);
        EXPECT_EQ(a(1), 1000);

        auto v2 = dynamic_view(a, slice_vector({range(3, 5)}));
        t b = {-100, -100};
        v2 = b;
        EXPECT_EQ(v2(1), -100);
        EXPECT_EQ(a(4), -100);
    }

    TEST(xdynamic_view, ellipsis)
    {
        using t = xarray<int>;
        auto a = t::from_shape({5, 5, 1, 1, 1, 4});
        std::iota(a.begin(), a.end(), 0);

        auto v1 = dynamic_view(a, {1, 1, xt::ellipsis()});
        dynamic_shape<std::size_t> v1_s{1, 1, 1, 4};
        EXPECT_EQ(v1.shape(), v1_s);

        EXPECT_EQ(dynamic_view(a, {2, 2, xt::ellipsis()}), dynamic_view(a, {2, 2, xt::all(), xt::all(), xt::all(), xt::all()}));
        EXPECT_EQ(dynamic_view(a, {2, xt::ellipsis(), 0, 2}), dynamic_view(a, {2, xt::all(), xt::all(), xt::all(), 0, 2}));

        EXPECT_THROW(dynamic_view(a, {xt::ellipsis(), 0, xt::ellipsis()}), std::runtime_error);

        t b = xt::ones<int>({5, 5, 5});
        auto v2 = dynamic_view(b, {xt::ellipsis(), 1, 1, 1});
        EXPECT_EQ(v2(), 1);
        EXPECT_EQ(v2.shape().size(), 0);

        auto v3 = dynamic_view(b, {xt::ellipsis(), 1, xt::all(), 1});
        dynamic_shape<std::size_t> v3_s{5};
        EXPECT_EQ(v3.shape(), v3_s);


        EXPECT_THROW(dynamic_view(b, {xt::ellipsis(), 1, 1, 1, 1}), std::runtime_error);
    }

    TEST(xdynamic_view, incompatible_shape)
    {
        xarray<int> a = xarray<int>::from_shape({ 4, 3, 2 });
        xarray<int> b = xarray<int>::from_shape({ 2, 3, 4 });
        slice_vector sv;
        sv.push_back(all());
        auto v = dynamic_view(a, sv);

        EXPECT_FALSE(broadcastable(v.shape(), b.shape()));
        EXPECT_FALSE(broadcastable(b.shape(), v.shape()));
        EXPECT_THROW(assert_compatible_shape(b, v), broadcast_error);
        EXPECT_THROW(assert_compatible_shape(v, b), broadcast_error);
        EXPECT_THROW(v = b, broadcast_error);
        EXPECT_THROW(noalias(v) = b, broadcast_error);
    }

    TEST(xdynamic_view, range_integer_casting)
    {
        // just check compilation
        auto arr = xarray<int>::from_shape({3, 4, 5});
        auto a = dynamic_view(arr, {range(0, std::ptrdiff_t(2)), 323});
        auto b = dynamic_view(arr, {range(std::size_t(0), 2), 323});
    }
}
