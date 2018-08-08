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
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;
    using view_shape_type = dynamic_shape<size_t>;

    bool operator==(const view_shape_type& lhs, const dynamic_shape<ptrdiff_t>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    bool operator==(const dynamic_shape<ptrdiff_t>& lhs, const dynamic_shape<size_t>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    TEST(xstrided_view, simple)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = strided_view(a, xstrided_slice_vector({ 1, range(1, 4) }));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_EQ(view1(), view1(0));
        EXPECT_EQ(view1(1, 0), view1(0));
        EXPECT_EQ(view1.unchecked(0), view1(0));

        auto view0 = strided_view(a, xstrided_slice_vector({ 0, range(0, 3) }));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(size_t(1), view0.dimension());
        EXPECT_EQ(size_t(3), view0.shape()[0]);

        auto view2 = strided_view(a, xstrided_slice_vector({ range(0, 2), 2 }));
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(2), view2.shape()[0]);

        auto view4 = strided_view(a, { 1 });
        EXPECT_EQ(size_t(1), view4.dimension());
        EXPECT_EQ(size_t(4), view4.shape()[0]);

        auto view5 = strided_view(view4, { 1 });
        EXPECT_EQ(size_t(0), view5.dimension());
        EXPECT_EQ(size_t(0), view5.shape().size());

        auto view6 = strided_view(a, xstrided_slice_vector({ 1, all() }));
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = strided_view(a, xstrided_slice_vector({ all(), 2 }));
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));

        EXPECT_THROW(strided_view(a, { all(), all(), 1 }), std::runtime_error);
        EXPECT_THROW(strided_view(a, { all(), all(), all() }), std::runtime_error);
        EXPECT_NO_THROW(strided_view(a, { all(), newaxis(), all() }));
        EXPECT_NO_THROW(strided_view(a, { 3, newaxis(), 1 }));
        EXPECT_NO_THROW(strided_view(a, { 3, 1, newaxis() }));
    }

    TEST(xstrided_view, three_dimensional)
    {
        view_shape_type shape = { 3, 4, 2 };
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

        auto view1 = strided_view(a, { 1 });
        EXPECT_EQ(size_t(2), view1.dimension());
        view_shape_type expected_shape = { 4, 2 };
        EXPECT_EQ(expected_shape, view1.shape());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));

        std::array<std::size_t, 2> idx = { 1, 1 };
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xstrided_view, iterator)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ range(0, 2), 1, range(1, 4) }));
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

        auto view2 = strided_view(view1, xstrided_slice_vector({ range(0, 2), range(1, 3) }));
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

    TEST(xstrided_view, fill)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double> a(shape), res(shape);
        std::vector<double> data{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());
        std::vector<double> data_res = { 1, 2, 3, 4, 5, 4, 4, 4, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 4, 4, 4, 21, 22, 23, 24 };
        std::copy(data_res.cbegin(), data_res.cend(), res.template begin<layout_type::row_major>());
        auto view1 = strided_view(a, xstrided_slice_vector({ range(0, 2), 1, range(1, 4) }));
        view1.fill(4);
        EXPECT_EQ(a, res);
    }

    TEST(xstrided_view, xstrided_view_on_xfunction)
    {
        xarray<int> a = { { 1, 2, 3, 4 },{ 5, 6, 7, 8 },{ 9, 10, 11, 12 } };
        xarray<int> b = { 1, 2, 3, 4 };

        auto sum = a + b;
        auto func = strided_view(sum, xstrided_slice_vector({ 1, range(1, 4) }));
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

    TEST(xstrided_view, view_on_generator)
    {
        auto vgen = strided_view(eye(4), xstrided_slice_vector({ 1, range(1, 4) }));
        auto iter = vgen.cbegin();
    }

    TEST(xstrided_view, xstrided_view_on_xtensor)
    {
        xtensor<int, 2> a({ 3, 4 });
        std::vector<int> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ 1, range(1, 4) }));
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

        xarray<int> b({ 3 }, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xstrided_view, const_view)
    {
        const xtensor<double, 3> arr{ { 1, 2, 3 }, 2.5 };
        xtensor<double, 2> arr2{ { 2, 3 }, 0.0 };
        xtensor<double, 2> ref{ { 2, 3 }, 2.5 };
        arr2 = strided_view(arr, { 0 });
        EXPECT_EQ(ref, arr2);
        // check that the following compiles
        auto v = strided_view(arr, { 0 });
        double acc = v(0);
        EXPECT_EQ(acc, 2.5);
        auto iter = v.begin();
    }

    TEST(xstrided_view, newaxis)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = strided_view(a, xstrided_slice_vector({ all(), newaxis(), all() }));
        EXPECT_EQ(a(1, 1), view1(1, 0, 1));
        EXPECT_EQ(a(1, 2), view1(1, 0, 2));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape()[0]);
        EXPECT_EQ(size_t(1), view1.shape()[1]);
        EXPECT_EQ(size_t(4), view1.shape()[2]);
        EXPECT_EQ(view1(0, 1), view1(0, 0, 1));
        EXPECT_EQ(view1(2, 1, 0, 1), view1(1, 0, 1));

        auto view2 = strided_view(a, xstrided_slice_vector({ all(), all(), newaxis() }));
        EXPECT_EQ(a(1, 1), view2(1, 1, 0));
        EXPECT_EQ(a(1, 2), view2(1, 2, 0));
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape()[0]);
        EXPECT_EQ(size_t(4), view2.shape()[1]);
        EXPECT_EQ(size_t(1), view2.shape()[2]);

        auto view3 = strided_view(a, xstrided_slice_vector({ 1, newaxis(), all() }));
        EXPECT_EQ(a(1, 1), view3(0, 1));
        EXPECT_EQ(a(1, 2), view3(0, 2));
        EXPECT_EQ(size_t(2), view3.dimension());

        auto view4 = strided_view(a, xstrided_slice_vector({ 1, all(), newaxis() }));
        EXPECT_EQ(a(1, 1), view4(1, 0));
        EXPECT_EQ(a(1, 2), view4(2, 0));
        EXPECT_EQ(size_t(2), view4.dimension());

        auto view5 = strided_view(view1, { 1 });
        EXPECT_EQ(a(1, 1), view5(0, 1));
        EXPECT_EQ(a(1, 2), view5(0, 2));
        EXPECT_EQ(size_t(2), view5.dimension());

        auto view6 = strided_view(view2, { 1 });
        EXPECT_EQ(a(1, 1), view6(1, 0));
        EXPECT_EQ(a(1, 2), view6(2, 0));
        EXPECT_EQ(size_t(2), view6.dimension());

        std::array<std::size_t, 3> idx1 = { 1, 0, 2 };
        EXPECT_EQ(a(1, 2), view1.element(idx1.begin(), idx1.end()));

        std::array<std::size_t, 3> idx2 = { 1, 2, 0 };
        EXPECT_EQ(a(1, 2), view2.element(idx2.begin(), idx2.end()));
    }

    TEST(xstrided_view, newaxis_iterating)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ all(), all(), newaxis() }));
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

        auto view2 = strided_view(a, xstrided_slice_vector({ all(), newaxis(), all() }));
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

    TEST(xstrided_view, newaxis_function)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        xarray<double> b(view_shape_type(1, 4));
        auto data_end = data.cbegin();
        data_end += 4;
        std::copy(data.cbegin(), data_end, b.template begin<layout_type::row_major>());

        auto v = strided_view(b, xstrided_slice_vector({ newaxis(), all() }));
        xarray<double> res = a + v;

        std::vector<double> data2{ 2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16 };
        xarray<double> expected(shape);
        std::copy(data2.cbegin(), data2.cend(), expected.template begin<layout_type::row_major>());

        EXPECT_EQ(expected, res);
    }

    TEST(xstrided_view, range_adaptor)
    {
        using namespace xt::placeholders;
        using t = xarray<int>;
        t a = { 1, 2, 3, 4, 5 };

        auto n = xnone();

        auto v1 = strided_view(a, xstrided_slice_vector({ range(3, _) }));
        t v1e = { 4, 5 };
        EXPECT_TRUE(v1e == v1);

        auto v2 = strided_view(a, xstrided_slice_vector({ range(_, 2) }));
        t v2e = { 1, 2 };
        EXPECT_TRUE(v2e == v2);

        auto v3 = strided_view(a, xstrided_slice_vector({ range(n, n) }));
        t v3e = { 1, 2, 3, 4, 5 };
        EXPECT_TRUE(v3e == v3);

        auto v4 = strided_view(a, xstrided_slice_vector({ range(n, 2, -1) }));
        t v4e = { 5, 4 };
        EXPECT_TRUE(v4e == v4);

        auto v5 = strided_view(a, xstrided_slice_vector({ range(2, n, -1) }));
        t v5e = { 3, 2, 1 };
        EXPECT_TRUE(v5e == v5);

        auto v6 = strided_view(a, xstrided_slice_vector({ range(n, n, n) }));
        t v6e = { 1, 2, 3, 4, 5 };
        EXPECT_TRUE(v6e == v6);

        auto v7 = strided_view(a, xstrided_slice_vector({ range(1, n, 2) }));
        t v7e = { 2, 4 };
        EXPECT_TRUE(v7e == v7);

        auto v8 = strided_view(a, xstrided_slice_vector({ range(2, n, 2) }));
        t v8e = { 3, 5 };
        EXPECT_TRUE(v8e == v8);
    }

    TEST(xstrided_view, assign)
    {
        using t = xarray<int>;
        t a = { 1, 2, 3, 4, 5 };

        auto v = strided_view(a, xstrided_slice_vector({ range(0, 2) }));
        v = 1000;
        EXPECT_EQ(v(0), 1000);
        EXPECT_EQ(a(0), 1000);
        EXPECT_EQ(a(1), 1000);

        auto v2 = strided_view(a, xstrided_slice_vector({ range(3, 5) }));
        t b = { -100, -100 };
        v2 = b;
        EXPECT_EQ(v2(1), -100);
        EXPECT_EQ(a(4), -100);
    }

    TEST(xstrided_view, ellipsis)
    {
        using t = xarray<int>;
        auto a = t::from_shape({ 5, 5, 1, 1, 1, 4 });
        std::iota(a.begin(), a.end(), 0);

        auto v1 = strided_view(a, { 1, 1, xt::ellipsis() });
        dynamic_shape<std::size_t> v1_s{ 1, 1, 1, 4 };
        EXPECT_EQ(v1.shape(), v1_s);

        EXPECT_EQ(strided_view(a, { 2, 2, xt::ellipsis() }), strided_view(a, { 2, 2, xt::all(), xt::all(), xt::all(), xt::all() }));
        EXPECT_EQ(strided_view(a, { 2, xt::ellipsis(), 0, 2 }), strided_view(a, { 2, xt::all(), xt::all(), xt::all(), 0, 2 }));

        EXPECT_THROW(strided_view(a, { xt::ellipsis(), 0, xt::ellipsis() }), std::runtime_error);

        t b = xt::ones<int>({ 5, 5, 5 });
        auto v2 = strided_view(b, { xt::ellipsis(), 1, 1, 1 });
        EXPECT_EQ(v2(), 1);
        EXPECT_EQ(v2.shape().size(), 0);

        auto v3 = strided_view(b, { xt::ellipsis(), 1, xt::all(), 1 });
        dynamic_shape<std::size_t> v3_s{ 5 };
        EXPECT_EQ(v3.shape(), v3_s);


        EXPECT_THROW(strided_view(b, { xt::ellipsis(), 1, 1, 1, 1 }), std::runtime_error);
    }

    TEST(xstrided_view, incompatible_shape)
    {
        xarray<int> a = xarray<int>::from_shape({ 4, 3, 2 });
        xarray<int> b = xarray<int>::from_shape({ 2, 3, 4 });
        xstrided_slice_vector sv;
        sv.push_back(all());
        auto v = strided_view(a, sv);

        EXPECT_FALSE(broadcastable(v.shape(), b.shape()));
        EXPECT_FALSE(broadcastable(b.shape(), v.shape()));
        EXPECT_THROW(assert_compatible_shape(b, v), broadcast_error);
        EXPECT_THROW(assert_compatible_shape(v, b), broadcast_error);
        EXPECT_THROW(v = b, broadcast_error);
        EXPECT_THROW(noalias(v) = b, broadcast_error);
    }

    TEST(xstrided_view, view_on_view)
    {
        xarray<int> a = xt::ones<int>({ 3, 4, 5 });
        auto v1 = strided_view(a, { 1, all(), all() });
        auto vv1 = strided_view(v1, { 1, all() });
        vv1 = vv1 * 5;
        EXPECT_EQ(a(0, 0, 0), 1);
        EXPECT_EQ(a(1, 1, 0), 5);
        EXPECT_EQ(a(1, 1, 4), 5);
        EXPECT_EQ(a(1, 2, 4), 1);
        EXPECT_EQ(v1(1, 4), 5);

        bool st = std::is_same<decltype(v1), decltype(vv1)>::value;
        EXPECT_TRUE(st);

        a = xt::ones<int>({ 3, 4, 5 });
        auto v2 = strided_view(a, { all(), 1, all() });
        auto vv2 = strided_view(v2, { all(), 2 });
        vv2 = vv2 * 5;
        EXPECT_EQ(a(0, 0, 0), 1);
        EXPECT_EQ(a(1, 1, 2), 5);
        EXPECT_EQ(a(2, 1, 2), 5);
        EXPECT_EQ(a(0, 1, 2), 5);
        EXPECT_EQ(v2(0, 2), 5);
        EXPECT_TRUE(xt::all(equal(vv2, 5)));
    }

    TEST(xstrided_view, range_integer_casting)
    {
        // just check compilation
        auto arr = xarray<int>::from_shape({ 3, 4, 5 });
        auto a = strided_view(arr, { range(0, std::ptrdiff_t(2)), 323 });
        auto b = strided_view(arr, { range(std::size_t(0), 2), 323 });
    }

    TEST(xstrided_view, strides)
    {
        // Strides: 72/24/6/1
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({ 5, 3, 4, 6 });

        auto s1 = strided_view(a, { 1, 1, xt::all(), xt::all() }).strides();
        std::vector<std::size_t> s1e = { 6, 1 };
        EXPECT_EQ(s1, s1e);

        auto s2 = strided_view(a, { 1, xt::all(), xt::all(), 1 }).strides();
        std::vector<std::size_t> s2e = { 24, 6 };
        EXPECT_EQ(s2, s2e);

        auto s3 = strided_view(a, { 1, xt::all(), 1, xt::newaxis(), xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s3e = { 24, 0, 0, 1 };
        EXPECT_EQ(s3, s3e);

        auto s4 = strided_view(a, { xt::range(0, 1, 2), 1, 0, xt::all(), xt::newaxis() }).strides();
        std::vector<std::size_t> s4e = { 0, 1, 0 };
        EXPECT_EQ(s4, s4e);

        auto s4x = strided_view(a, { xt::range(0, 5, 2), 1, 0, xt::all(), xt::newaxis() }).strides();
        std::vector<std::size_t> s4xe = { 72 * 2, 1, 0 };
        EXPECT_EQ(s4x, s4xe);

        auto s5 = strided_view(a, { xt::all(), 1 }).strides();
        std::vector<std::size_t> s5e = { 72, 6, 1 };
        EXPECT_EQ(s5, s5e);

        auto s6 = strided_view(a, { xt::all(), 1, 1, xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s6e = { 72, 0, 1 };
        EXPECT_EQ(s6, s6e);

        auto s7 = strided_view(a, { xt::all(), 1, xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s7e = { 72, 0, 6, 1 };
        EXPECT_EQ(s7, s7e);
    }

    TEST(xstrided_view, layout)
    {
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({ 5, 3, 4, 6 });

        auto s1 = strided_view(a, { xt::all(), 1, xt::newaxis(), xt::all() }).layout();
        EXPECT_EQ(s1, layout_type::dynamic);

        auto s1x = strided_view(a, { 1, xt::all(), xt::newaxis(), xt::all() }).layout();
        EXPECT_EQ(s1x, layout_type::row_major);

        auto s2 = strided_view(a, { 1, 2, range(0, 3), xt::all() }).layout();
        EXPECT_EQ(s2, layout_type::row_major);

        auto s3 = strided_view(a, { 1 }).layout();
        EXPECT_EQ(s3, layout_type::row_major);

        xarray<int, layout_type::column_major> b = xarray<int, layout_type::column_major>::from_shape({ 5, 3, 4, 6 });

        auto s4 = strided_view(b, { 1 }).layout();
        EXPECT_EQ(s4, layout_type::dynamic);

        auto s5 = strided_view(b, { xt::all(), 1, 1, 1 }).layout();
        EXPECT_EQ(s5, layout_type::column_major);

        auto s6 = strided_view(b, { xt::all(), 1, 1, xt::range(0, 6) }).layout();
        EXPECT_EQ(s6, layout_type::dynamic);
    }

    TEST(xstrided_view, transpose_assignment)
    {
        xarray<double> e = xt::arange<double>(24);
        e.resize({2, 2, 6});
        auto vt = transpose(e);

        vt(0, 0, 1) = 123;
        EXPECT_EQ(123, e(1, 0, 0));
        auto val = vt[{1, 0, 1}];
        EXPECT_EQ(e(1, 0, 1), val);
        EXPECT_ANY_THROW(vt.at(10, 10, 10));
        EXPECT_ANY_THROW(vt.at(0, 0, 0, 0));
    }

    TEST(xstrided_view, expression_adapter)
    {
        auto e = xt::arange<double>(24);
        auto sv = xstrided_slice_vector({range(2, 10, 3)});
        auto vt = strided_view(e, sv);

        EXPECT_EQ(vt(0), 2);
        EXPECT_EQ(vt(1), 5);

        xt::xarray<double> assigned = vt;
        EXPECT_EQ(assigned, vt);
        EXPECT_EQ(assigned(1), 5);
    }

    TEST(xstrided_view, transpose_layout_swap)
    {
        xarray<double, layout_type::row_major> a = xt::ones<double>({5, 5});

        auto tv = transpose(a);
        EXPECT_EQ(tv.layout(), layout_type::column_major);

        auto tvt = transpose(tv);
        EXPECT_EQ(tvt.layout(), layout_type::row_major);

        xarray<double, layout_type::column_major> b = xt::ones<double>({5, 5, 5});
        auto cbt = transpose(b);
        EXPECT_EQ(cbt.layout(), layout_type::row_major);

        auto cbw1 = transpose(b, {0, 1 ,2});
        auto cbw2 = transpose(b, {2, 1, 0});
        auto cbw3 = transpose(b, {2, 0, 1});
        EXPECT_EQ(cbw1.layout(), layout_type::column_major);
        EXPECT_EQ(cbw2.layout(), layout_type::row_major);
        EXPECT_EQ(cbw3.layout(), layout_type::dynamic);
    }

    TEST(xstrided_view, transpose_function)
    {
        xarray<int, layout_type::row_major> a = { { 0, 1, 2 }, { 3, 4, 5 } };
        xarray<int, layout_type::row_major> b = { { 0, 1, 2 }, { 3, 4, 5 } };
        auto fun = a + b;
        auto tr = transpose(fun);
        EXPECT_EQ(fun(0, 0), tr(0, 0));
        EXPECT_EQ(fun(0, 1), tr(1, 0));
        EXPECT_EQ(fun(0, 2), tr(2, 0));
        EXPECT_EQ(fun(1, 0), tr(0, 1));
        EXPECT_EQ(fun(1, 1), tr(1, 1));
        EXPECT_EQ(fun(1, 2), tr(2, 1));

        xarray<int, layout_type::column_major> a2 = { { 0, 1, 2 },{ 3, 4, 5 } };
        xarray<int, layout_type::column_major> b2 = { { 0, 1, 2 },{ 3, 4, 5 } };
        auto fun2 = a2 + b2;
        auto tr2 = transpose(fun2);
        EXPECT_EQ(fun2(0, 0), tr2(0, 0));
        EXPECT_EQ(fun2(0, 1), tr2(1, 0));
        EXPECT_EQ(fun2(0, 2), tr2(2, 0));
        EXPECT_EQ(fun2(1, 0), tr2(0, 1));
        EXPECT_EQ(fun2(1, 1), tr2(1, 1));
        EXPECT_EQ(fun2(1, 2), tr2(2, 1));
    }

    TEST(xstrided_view, ravel)
    {
        xarray<int, layout_type::row_major> a = { { 0, 1, 2 },{ 3, 4, 5 } };

        auto flat = ravel<layout_type::row_major>(a);
        EXPECT_EQ(flat(0), a(0, 0));
        EXPECT_EQ(flat(1), a(0, 1));
        EXPECT_EQ(flat(2), a(0, 2));
        EXPECT_EQ(flat(3), a(1, 0));
        EXPECT_EQ(flat(4), a(1, 1));
        EXPECT_EQ(flat(5), a(1, 2));

        auto flat_c = ravel<layout_type::column_major>(a);
        EXPECT_EQ(flat_c(0), a(0, 0));
        EXPECT_EQ(flat_c(1), a(1, 0));
        EXPECT_EQ(flat_c(2), a(0, 1));
        EXPECT_EQ(flat_c(3), a(1, 1));
        EXPECT_EQ(flat_c(4), a(0, 2));
        EXPECT_EQ(flat_c(5), a(1, 2));

        auto flat2 = flatten(a);
        EXPECT_EQ(flat, flat2);
    }

    TEST(xstrided_view, split)
    {
        auto b = xt::xarray<double>::from_shape({3, 3, 3});
        using ds = xt::dynamic_shape<std::size_t>;
        std::iota(b.begin(), b.end(), 0);
        auto s1 = split(b, 3);
        EXPECT_EQ(s1.size(), 3u);
        EXPECT_EQ(s1[0].shape(), ds({1, 3, 3}));
        EXPECT_EQ(s1[0](0, 0), b(0, 0, 0));
        EXPECT_EQ(s1[1](0, 0), b(1, 0, 0));
        EXPECT_EQ(s1[2](0, 0), b(2, 0, 0));

        EXPECT_THROW(split(b, 4), std::runtime_error);
        EXPECT_THROW(split(b, 2), std::runtime_error);

        auto s2 = split(b, 1);
        EXPECT_EQ(s2.size(), 1u);
        EXPECT_EQ(s2[0].shape(), ds({3, 3, 3}));

        auto s3 = split(b, 3, 1);
        EXPECT_EQ(s3.size(), 3);
        EXPECT_EQ(s3[0].shape(), ds({3, 1, 3}));

        EXPECT_EQ(s3[0](0, 1), b(0, 0, 1));
        EXPECT_EQ(s3[1](0, 1), b(0, 1, 1));
        EXPECT_EQ(s3[2](0, 1), b(0, 2, 1));
    }

    TEST(xstrided_view, squeeze)
    {
        auto b = xt::xarray<double>::from_shape({3, 3, 1, 1, 2, 1, 3});
        std::iota(b.begin(), b.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;
        auto sq = squeeze(b);

        EXPECT_EQ(sq.shape(), ds({3, 3, 2, 3}));
        EXPECT_EQ(sq(1, 1, 1, 1), b(1, 1, 0, 0, 1, 0, 1));
        EXPECT_THROW(squeeze(b, 1, check_policy::full()), std::runtime_error);
        EXPECT_THROW(squeeze(b, 10, check_policy::full()), std::runtime_error);

        auto sq2 = squeeze(b, {2, 3}, check_policy::full());
        EXPECT_EQ(sq2.shape(), ds({3, 3, 2, 1, 3}));
        EXPECT_EQ(sq2(1, 1, 1, 0, 1), b(1, 1, 0, 0, 1, 0, 1));

        auto sq3 = squeeze(b, 2);
        EXPECT_EQ(sq3.shape(), ds({3, 3, 1, 2, 1, 3}));
        EXPECT_EQ(sq3(2, 2, 0, 1, 0, 2), b(2, 2, 0, 0, 1, 0, 2));
    }

    TEST(xstrided_view, expand_dims)
    {
        auto b = xt::xarray<double>::from_shape({3, 3});
        std::iota(b.begin(), b.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;
        auto ex = expand_dims(b, 0);
        EXPECT_EQ(ex.shape(), ds({1, 3, 3}));
        auto ex1 = expand_dims(b, 1);
        EXPECT_EQ(ex1.shape(), ds({3, 1, 3}));
        auto ex2 = expand_dims(b, 2);
        EXPECT_EQ(ex2.shape(), ds({3, 3, 1}));

        EXPECT_EQ(ex1(0, 0, 1), b(0, 1));
        EXPECT_EQ(ex1(2, 0, 1), b(2, 1));
    }

    TEST(xstrided_view, atleast_nd)
    {
        xt::xarray<char> d0 = 123;
        auto d1 = xt::xarray<char>::from_shape({3});
        auto d2 = xt::xarray<char>::from_shape({3, 3});
        auto d3 = xt::xarray<char>::from_shape({3, 3, 3});
        auto d5 = xt::xarray<char>::from_shape({3, 3, 3, 3, 3});
        std::iota(d1.begin(), d1.end(), 0);
        std::iota(d2.begin(), d2.end(), 0);
        std::iota(d3.begin(), d3.end(), 0);
        std::iota(d5.begin(), d5.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;

        auto d3d1 = atleast_3d(d1);
        EXPECT_EQ(d3d1.shape(), ds({1, 3, 1}));
        auto d3d2 = atleast_3d(d2);
        EXPECT_EQ(d3d2.shape(), ds({3, 3, 1}));
        auto d3d3 = atleast_3d(d3);
        EXPECT_EQ(d3d3.shape(), ds({3, 3, 3}));
        auto d3d5 = atleast_3d(d5);
        EXPECT_EQ(d3d5.shape(), ds({3, 3, 3, 3, 3}));
        auto d3d0 = atleast_3d(d0);
        EXPECT_EQ(d3d0.shape(), ds({1, 1, 1}));
        EXPECT_EQ(d3d0(0, 0, 0), 123);
        auto d4d1 = atleast_Nd<4>(d1);
        EXPECT_EQ(d4d1.shape(), ds({1, 3, 1, 1}));
        auto d2d1 = atleast_2d(d1);
        EXPECT_EQ(d2d1.shape(), ds({1, 3}));
    }

    TEST(xstrided_view, trim_zeros)
    {
        using arr_t = xarray<int>;
        arr_t a = {0, 0, 0, 1, 3, 0};
        arr_t b = {0, 0, 0, 0};
        arr_t c = {0, 0, 0, 1};
        arr_t d = {1, 0, 0, 1};

        arr_t ea = {1, 3};
        arr_t ec = {1};
        arr_t ed = {1, 0, 0, 1};

        arr_t eaf = {1, 3, 0};
        arr_t ecf = {1};
        arr_t edf = {1, 0, 0, 1};

        arr_t eab = {0, 0, 0, 1, 3};
        arr_t ecb = {0, 0, 0, 1};
        arr_t edb = {1, 0, 0, 1};

        EXPECT_EQ(trim_zeros(a), ea);
        EXPECT_EQ(trim_zeros(b).size(), 0);
        EXPECT_EQ(trim_zeros(c), ec);
        EXPECT_EQ(trim_zeros(d), ed);

        EXPECT_EQ(trim_zeros(a, "f"), eaf);
        EXPECT_EQ(trim_zeros(b, "f").size(), 0);
        EXPECT_EQ(trim_zeros(c, "f"), ecf);
        EXPECT_EQ(trim_zeros(d, "f"), edf);

        EXPECT_EQ(trim_zeros(a, "b"), eab);
        EXPECT_EQ(trim_zeros(b, "b").size(), 0);
        EXPECT_EQ(trim_zeros(c, "b"), ecb);
        EXPECT_EQ(trim_zeros(d, "b"), edb);
    }

    TEST(xstrided_view, flipud)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 0);
        xarray<double> expected = {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(7, t[idx]);
        ASSERT_EQ(2, t(2, 1));
        ASSERT_EQ(7, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 0);
        xarray<double> expected_2 = {{{6, 7, 8},
                                      {9, 10, 11}},
                                     {{0, 1, 2},
                                      {3, 4, 5}}};
        ASSERT_EQ(expected_2, ft);
    }

    TEST(xstrided_view, fliplr)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 1);
        xarray<double> expected = {{3, 2, 1}, {6, 5, 4}, {9, 8, 7}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(3, t[idx]);
        ASSERT_EQ(8, t(2, 1));
        ASSERT_EQ(3, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 1);
        xarray<double> expected_2 = {{{3, 4, 5},
                                      {0, 1, 2}},
                                     {{9, 10, 11},
                                      {6, 7, 8}}};

        ASSERT_EQ(expected_2, ft);
        auto flipped_range = xt::flip(xt::stack(xt::xtuple(arange<double>(2), arange<double>(2))), 1);
        xarray<double> expected_range = {{1, 0}, {1, 0}};
        ASSERT_TRUE(all(equal(flipped_range, expected_range)));
    }

    TEST(xstrided_view, reverse_iteration)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, {range(0, 2), 1, range(1, 4)});
        auto iter = view1.template rbegin<layout_type::row_major>();
        auto iter_end = view1.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);

        auto view2 = strided_view(view1, {range(0, 2), range(1, 3)});
        auto iter2 = view2.template rbegin<layout_type::row_major>();
        auto iter_end2 = view2.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
    }

    TEST(xstrided_view, reshape_view)
    {
        xarray<double> a = xt::arange<double>(9);
        auto av = xt::reshape_view(a, {3, 3});
        auto xv = xt::reshape_view(a, xshape<3, 3>());

        xtensor<double, 2> e = xt::reshape_view(xt::arange(9), {3, 3});

        a.reshape({3, 3});
        EXPECT_EQ(av, e);
        EXPECT_EQ(av, a);

        bool truthy;
        truthy = std::is_same<typename decltype(xv)::temporary_type, xtensor_fixed<double, xshape<3, 3>>>();
        EXPECT_TRUE(truthy);

    #if !defined(X_OLD_CLANG)
        truthy = std::is_same<typename decltype(av)::temporary_type, xtensor<double, 2>>();
        EXPECT_TRUE(truthy);
        truthy = std::is_same<typename decltype(av)::shape_type, typename decltype(e)::shape_type>::value;
        EXPECT_TRUE(truthy);
    #endif
    }

    TEST(xstrided_view, on_xview)
    {
        xarray<double> a = {0,1,2,3,4,5,6,7,8};
        auto v = view(a, keep(3, 5, 6));
        auto v2 = strided_view(v, {2});
        EXPECT_EQ(v2(0), 6);
        auto v3 = strided_view(v, {all()});
        EXPECT_EQ(v3(0), 3);
        EXPECT_EQ(v3(1), 5);
    }
}
