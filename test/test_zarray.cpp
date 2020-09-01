/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/zarray.hpp"
#include "xtensor/zdispatcher.hpp"

#ifndef XTENSOR_DISABLE_EXCEPTIONS
namespace xt
{
    TEST(zarray, value_semantics)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> ra = {{2., 2.}, {3., 4.}};
        zarray da(a);
        da.get_array<double>()(0, 0) = 2.;

        EXPECT_EQ(a, ra);
    }

    // TODO : move to dedicated test file
    TEST(zarray, dispatching)
    {
        using dispatcher_type = zsingle_dispatcher<math::exp_fun>;
        dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> expa = {{std::exp(0.5), std::exp(1.5)}, {std::exp(2.5), std::exp(3.5)}};
        xarray<double> res;
        zarray za(a);
        zarray zres(res);

        dispatcher_type::dispatch(za.get_implementation(), zres.get_implementation());

        EXPECT_EQ(expa, res);
    }
}
#endif

