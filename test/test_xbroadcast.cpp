/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xbroadcast, broadcast)
    {
        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};

        auto m1_broadcast = broadcast(m1, {1, 2, 3});
        ASSERT_EQ(1.0, m1_broadcast(0, 0, 0));
        ASSERT_EQ(4.0, m1_broadcast(0, 1, 0));
        ASSERT_EQ(5.0, m1_broadcast(0, 1, 1));

        auto shape = std::vector<std::size_t> {1, 2, 3};
        auto m1_broadcast2 = broadcast(m1, shape);
        ASSERT_EQ(1.0, m1_broadcast2(0, 0, 0));
        ASSERT_EQ(4.0, m1_broadcast2(0, 1, 0));
        ASSERT_EQ(5.0, m1_broadcast2(0, 1, 1));

        double f = *(m1_broadcast.begin());
        xarray<double> m1_assigned = m1_broadcast;
        ASSERT_EQ(5.0, m1_assigned(0, 1, 1));
    }

    TEST(xbroadcast, element)
    {
        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};

        auto m1_broadcast = broadcast(m1, {4, 2, 3});

        // access with the right number of arguments
        std::array<std::size_t, 3> index1 = {0, 1, 1};
        ASSERT_EQ(5.0, m1_broadcast.element(index1.begin(), index1.end()));
        // too many arguments = using the last ones only
        std::array<std::size_t, 4> index3 = {4, 0, 1, 1};
        ASSERT_EQ(5.0, m1_broadcast.element(index3.begin(), index3.end()));
    }

    TEST(xbroadcast, shape_forwarding)
    {
        std::array<std::size_t, 2> bc_shape;
        auto m1_broadcast = broadcast(123, bc_shape);
    }
}

