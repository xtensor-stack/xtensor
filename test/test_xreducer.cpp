/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xreducer.hpp"

namespace xt
{
    TEST(xreducer, basic)
    {
        std::array<std::size_t, 2> axis = { 1, 3 };
        xarray<double> a = ones<double>({ 3, 2, 4, 6, 5 });

        using func = std::plus<double>;
        xreducer<func, const xarray<double>&, 2> red(func(), a, axis);

        EXPECT_EQ(12, red(0, 0, 0));
    }
}