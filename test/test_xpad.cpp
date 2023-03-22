/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <complex>
#include <limits>

#include "xtensor/xio.hpp"
#include "xtensor/xpad.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(xpad, constant_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 2, 0, 0, 0},
            {0, 0, 0, 3, 4, 5, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2, 2}, {3, 3}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 0}, {3, 4, 5, 0, 0}, {0, 0, 0, 0, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0, 1}, {0, 2}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 0, 0, 0, 0}, {0, 0, 0, 1, 2}, {0, 0, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1, 0}, {2, 0}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {
            {0, 1, 2, 0, 1, 2, 0, 1, 2},
            {3, 4, 5, 3, 4, 5, 3, 4, 5},
            {0, 1, 2, 0, 1, 2, 0, 1, 2},
            {3, 4, 5, 3, 4, 5, 3, 4, 5},
            {0, 1, 2, 0, 1, 2, 0, 1, 2},
            {3, 4, 5, 3, 4, 5, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2, 2}, {3, 3}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 1}, {3, 4, 5, 3, 4}, {0, 1, 2, 0, 1}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0, 1}, {0, 2}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{4, 5, 3, 4, 5}, {1, 2, 0, 1, 2}, {4, 5, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1, 0}, {2, 0}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {
            {5, 4, 3, 3, 4, 5, 5, 4, 3},
            {2, 1, 0, 0, 1, 2, 2, 1, 0},
            {2, 1, 0, 0, 1, 2, 2, 1, 0},
            {5, 4, 3, 3, 4, 5, 5, 4, 3},
            {5, 4, 3, 3, 4, 5, 5, 4, 3},
            {2, 1, 0, 0, 1, 2, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2, 2}, {3, 3}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 2, 1}, {3, 4, 5, 5, 4}, {3, 4, 5, 5, 4}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0, 1}, {0, 2}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{1, 0, 0, 1, 2}, {1, 0, 0, 1, 2}, {4, 3, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1, 0}, {2, 0}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {
            {5, 4, 3, 4, 5, 4, 3},
            {2, 1, 0, 1, 2, 1, 0},
            {5, 4, 3, 4, 5, 4, 3},
            {2, 1, 0, 1, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1, 1}, {2, 2}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 1, 0}, {3, 4, 5, 4, 3}, {0, 1, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0, 1}, {0, 2}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2}, {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{5, 4, 3, 4, 5}, {2, 1, 0, 1, 2}, {5, 4, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1, 0}, {2, 0}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_a)
    {
        xt::xtensor<size_t, 1> a = xt::arange<size_t>(3);

        xt::xtensor<size_t, 1> b = {0, 1, 2, 0, 1, 2, 0, 1, 2};

        xt::xtensor<size_t, 1> c = xt::tile(a, 3u);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_b)
    {
        xt::xtensor<size_t, 2> a = xt::arange<size_t>(3 * 2).reshape({3, 2});

        xt::xtensor<size_t, 2> b = {{0, 1}, {2, 3}, {4, 5}, {0, 1}, {2, 3}, {4, 5}, {0, 1}, {2, 3}, {4, 5}};

        xt::xtensor<size_t, 2> c = xt::tile(a, 3u);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_c)
    {
        xt::xtensor<size_t, 2> a = xt::arange<size_t>(3 * 2).reshape({3, 2});

        xt::xtensor<size_t, 2> b = {
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::tile(a, {3u, 4u});

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_d)
    {
        xt::xtensor<size_t, 2> a = xt::arange<size_t>(3 * 2).reshape({3, 2});

        xt::xtensor<size_t, 2> b = {
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::tile(a, std::array<size_t, 2>{{3u, 4u}});

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_e)
    {
        xt::xarray<size_t> a = xt::arange<size_t>(3 * 2).reshape({3, 2});

        xt::xtensor<size_t, 2> b = {
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5},
            {0, 1, 0, 1, 0, 1, 0, 1},
            {2, 3, 2, 3, 2, 3, 2, 3},
            {4, 5, 4, 5, 4, 5, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::tile(a, std::vector<size_t>{{3u, 4u}});

        EXPECT_EQ(b, c);
    }

    TEST(xpad, tile_f)
    {
        xt::xarray<double> m = xt::tile(xt::expand_dims(xt::eye<double>(4), 0), 3u);
        xt::xarray<double> expected = {
            {{1., 0., 0., 0.}, {0., 1., 0., 0.}, {0., 0., 1., 0.}, {0., 0., 0., 1.}},
            {{1., 0., 0., 0.}, {0., 1., 0., 0.}, {0., 0., 1., 0.}, {0., 0., 0., 1.}},
            {{1., 0., 0., 0.}, {0., 1., 0., 0.}, {0., 0., 1., 0.}, {0., 0., 0., 1.}}};
        EXPECT_EQ(m, expected);
    }
}
