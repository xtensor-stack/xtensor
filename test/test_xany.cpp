/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xany.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    TEST(xany, assignment)
    {
        xt::xarray<double> arr
          {{1.0, 2.0, 3.0},
           {2.0, 5.0, 7.0},
           {2.0, 5.0, 7.0}};

        xt::xtensor<double, 1> tens
          {1.0, 2.0, 3.0};

        xt::xany holder = arr;

        auto& larray = holder.get<xt::xarray<double>>();
        ASSERT_EQ(larray(1, 0), 2.0);
        std::cout << ",,," << std::endl;

        holder = tens;
        auto& ltensor = holder.get<xt::xtensor<double, 1>>();
        ASSERT_EQ(ltensor(2), 3.0);
    }
}
