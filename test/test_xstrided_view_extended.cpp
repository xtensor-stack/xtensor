/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// This file is generated from test/files/cppy_source/test_xstrided_view_extended.cppy by preprocess.py!

#include <algorithm>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    using namespace xt::placeholders;

    /*py
    a = np.arange(35).reshape(5, 7)
    */
    TEST(xstrided_view_extended, negative_slices_twod)
    {
        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
            xt::xarray<double> a = xt::arange(35);
            a.reshape({ 5, 7 });
            // py_av0 = a[:-2, ::-1]
            xarray<long> py_av0 = { { 6, 5, 4, 3, 2, 1, 0},
                                   {13,12,11,10, 9, 8, 7},
                                   {20,19,18,17,16,15,14} };
            auto av0 = xt::strided_view(a, { _r | _ | -2, _r | _ | _ | -1 });
            EXPECT_EQ(av0, py_av0);
            // py_av1 = a[::-1, ::-1]
            xarray<long> py_av1 = { {34,33,32,31,30,29,28},
                                   {27,26,25,24,23,22,21},
                                   {20,19,18,17,16,15,14},
                                   {13,12,11,10, 9, 8, 7},
                                   { 6, 5, 4, 3, 2, 1, 0} };
            auto av1 = xt::strided_view(a, { _r | _ | _ | -1, _r | _ | _ | -1 });
            EXPECT_EQ(av1, py_av1);
            // py_av2 = a[1:-3, -3:2:-1]
            xarray<long> py_av2 = { {11,10} };
            auto av2 = xt::strided_view(a, { _r | 1 | -3, _r | -3 | 2 | -1 });
            EXPECT_EQ(av2, py_av2);
            // py_av3 = a[-1:-4:-1, -3:1:-2]
            xarray<long> py_av3 = { {32,30},
                                   {25,23},
                                   {18,16} };
            auto av3 = xt::strided_view(a, { _r | -1 | -4 | -1, _r | -3 | 1 | -2 });
            EXPECT_EQ(av3, py_av3);
            auto av4 = xt::strided_view(a, { _r | -3 | -5, _r | -3 | 10 });
            EXPECT_EQ(av4.size(), 0);
            // py_av5 = a[-5:-2, -3:10]
            xarray<long> py_av5 = { { 4, 5, 6},
                                   {11,12,13},
                                   {18,19,20} };
            auto av5 = xt::strided_view(a, { _r | -5 | -2, _r | -3 | 10 });
            EXPECT_EQ(av5, py_av5);
        }
    }

    /*py
    a = np.arange(35).reshape(5, 7)
    a[0:-2] += a[:3:-1]
    at = np.copy(a)
    at[::-2] += at[::2]
    */
    TEST(xstrided_view_extended, negative_slices_math)
    {
        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
            xt::xarray<double> a = xt::arange(35);
            a.reshape({ 5, 7 });
            strided_view(a, { _r | 0 | -2 }) += strided_view(a, { {_r | _ | 3 | -1} });
            // py_a
            xarray<long> py_a = { {28,30,32,34,36,38,40},
                                 {35,37,39,41,43,45,47},
                                 {42,44,46,48,50,52,54},
                                 {21,22,23,24,25,26,27},
                                 {28,29,30,31,32,33,34} };
            EXPECT_EQ(a, py_a);
            strided_view(a, { _r | _ | _ | -2 }) += strided_view(a, { _r | _ | _ | 2 });
            // py_at
            xarray<long> py_at = { { 56, 59, 62, 65, 68, 71, 74},
                                  { 35, 37, 39, 41, 43, 45, 47},
                                  { 84, 88, 92, 96,100,104,108},
                                  { 21, 22, 23, 24, 25, 26, 27},
                                  { 56, 59, 62, 65, 68, 71, 74} };
            EXPECT_EQ(a, py_at);
        }
    }
}
