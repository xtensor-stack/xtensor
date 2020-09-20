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

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
// #include "xtensor/xfixed.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    template <class E, std::enable_if_t<!xt::has_rank_t<E, 2>::value, int> = 0>
    inline size_t sfinae_rank_basic_func(E&&)
    {
        return 0;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
    inline size_t sfinae_rank_basic_func(E&&)
    {
        return 2;
    }

    TEST(sfinae, rank_basic)
    {
        xt::xarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};
        // xt::xtensor_fixed<size_t, xt::xshape<2, 2>> d = {{9, 9}, {9, 9}};
        auto v = xt::view(c, 0, xt::all());

        EXPECT_TRUE(sfinae_rank_basic_func(a) == 0ul);
        EXPECT_TRUE(sfinae_rank_basic_func(b) == 0ul);
        EXPECT_TRUE(sfinae_rank_basic_func(c) == 2ul);
        // EXPECT_TRUE(sfinae_rank_basic_func(d) == 2ul);
        EXPECT_TRUE(sfinae_rank_basic_func(v) == 0ul);
        EXPECT_TRUE(sfinae_rank_basic_func(2ul * a) == 0ul);
        EXPECT_TRUE(sfinae_rank_basic_func(2ul * b) == 0ul);
        EXPECT_TRUE(sfinae_rank_basic_func(2ul * c) == 0ul);

    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, SIZE_MAX>::value, int> = 0>
    inline size_t sfinae_rank_func(E&&)
    {
        return 0;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 1>::value, int> = 0>
    inline size_t sfinae_rank_func(E&&)
    {
        return 1;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
    inline size_t sfinae_rank_func(E&&)
    {
        return 2;
    }

    TEST(sfinae, rank)
    {
        xt::xarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};
        // xt::xtensor_fixed<size_t, xt::xshape<2, 2>> d = {{9, 9}, {9, 9}};
        auto v = xt::view(c, 0, xt::all());

        EXPECT_TRUE(sfinae_rank_func(a) == 0ul);
        EXPECT_TRUE(sfinae_rank_func(b) == 1ul);
        EXPECT_TRUE(sfinae_rank_func(c) == 2ul);
        // EXPECT_TRUE(sfinae_rank_func(d) == 2ul);
        EXPECT_TRUE(sfinae_rank_func(v) == 0ul);
        EXPECT_TRUE(sfinae_rank_func(2ul * a) == 0ul);
        EXPECT_TRUE(sfinae_rank_func(2ul * b) == 0ul);
        EXPECT_TRUE(sfinae_rank_func(2ul * c) == 0ul);
    }

    template <class E, std::enable_if_t<!xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_fixed_func(E&&)
    {
        return false;
    }

    template <class E, std::enable_if_t<xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_fixed_func(E&&)
    {
        return true;
    }

    TEST(sfinae, fixed_rank)
    {
        xt::xarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};
        // xt::xtensor_fixed<size_t, xt::xshape<2, 2>> d = {{9, 9}, {9, 9}};
        auto v = xt::view(c, 0, xt::all());

        EXPECT_TRUE(sfinae_fixed_func(a) == false);
        EXPECT_TRUE(sfinae_fixed_func(b) == true);
        EXPECT_TRUE(sfinae_fixed_func(c) == true);
        // EXPECT_TRUE(sfinae_fixed_func(d) == 2ul);
        EXPECT_TRUE(sfinae_fixed_func(v) == false);
        EXPECT_TRUE(sfinae_fixed_func(2ul * a) == false);
        EXPECT_TRUE(sfinae_fixed_func(2ul * b) == false);
        EXPECT_TRUE(sfinae_fixed_func(2ul * c) == false);
    }

    template <class T>
    struct sfinae_get_rank
    {
        static const size_t rank = xt::get_rank<T>::value;

        static size_t value()
        {
            return rank;
        }
    };

    TEST(sfinae, get_rank)
    {
        xt::xtensor<double, 1> A = xt::zeros<double>({2});
        xt::xtensor<double, 2> B = xt::zeros<double>({2, 2});
        xt::xarray<double> C = xt::zeros<double>({2, 2});

        EXPECT_TRUE(sfinae_get_rank<decltype(A)>::value() == 1ul);
        EXPECT_TRUE(sfinae_get_rank<decltype(B)>::value() == 2ul);
        EXPECT_TRUE(sfinae_get_rank<decltype(C)>::value() == SIZE_MAX);
        EXPECT_TRUE(sfinae_get_rank<decltype(2.0 * A)>::value() == SIZE_MAX);
        EXPECT_TRUE(sfinae_get_rank<decltype(2.0 * B)>::value() == SIZE_MAX);
        EXPECT_TRUE(sfinae_get_rank<decltype(2.0 * C)>::value() == SIZE_MAX);
    }
}
