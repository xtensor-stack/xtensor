/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xstrided_view.hpp"

#include "xtensor/xio.hpp"

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

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
        auto sv = slice_vector({range(2, 10, 3)});
        auto vt = dynamic_view(e, sv);

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
        EXPECT_EQ(s1.size(), 3);
        EXPECT_EQ(s1[0].shape(), ds({1, 3, 3}));
        EXPECT_EQ(s1[0](0, 0), b(0, 0, 0));
        EXPECT_EQ(s1[1](0, 0), b(1, 0, 0));
        EXPECT_EQ(s1[2](0, 0), b(2, 0, 0));

        EXPECT_THROW(split(b, 4), std::runtime_error);
        EXPECT_THROW(split(b, 2), std::runtime_error);

        auto s2 = split(b, 1);
        EXPECT_EQ(s2.size(), 1);
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
}
