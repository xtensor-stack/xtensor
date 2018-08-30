/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xhistogram.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    TEST(xhistogram, histogram)
    {
        xt::xtensor<double,1> data = {1., 1., 2., 2.};

        {
            xt::xtensor<double,1> count = xt::histogram(data, std::size_t(2));

            EXPECT_EQ(count.size(), 2 );
            EXPECT_EQ(count[0]    , 2.);
            EXPECT_EQ(count[1]    , 2.);
        }

        {
            xt::xtensor<double,1> count = xt::histogram(data,
                xt::histogram_bin_edges(data, std::size_t(2), xt::histogram_algorithm::uniform)
            );

            EXPECT_EQ(count.size(), 2 );
            EXPECT_EQ(count[0]    , 2.);
            EXPECT_EQ(count[1]    , 2.);
        }
    }
}
