/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>
#include <sstream>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


namespace xt
{

    TEST(xio, one_d)
    {
        xarray<double> e{1, 2, 3, 4, 5};
        std::stringstream out;
        out << e;
        EXPECT_EQ("{1, 2, 3, 4, 5}", out.str());
    }

    TEST(xio, two_d)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(R"xio({{1, 2, 3, 4},
 {5, 6, 7, 8},
 {9, 10, 11, 12}})xio", out.str());
    }

    TEST(xio, three_d)
    {
        xarray<double> e{{{1, 2},
                          {3, 4},
                          {5, 6},
                          {7, 8}},
                         {{9, 10},
                          {11, 12},
                          {7, 9},
                          {11, 14}},
                         {{5, 26},
                          {7, 8},
                          {10, 8},
                          {4, 3}}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(R"xio({{{1, 2},
  {3, 4},
  {5, 6},
  {7, 8}},
 {{9, 10},
  {11, 12},
  {7, 9},
  {11, 14}},
 {{5, 26},
  {7, 8},
  {10, 8},
  {4, 3}}})xio", out.str());
    }
}

