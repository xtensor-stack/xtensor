/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xchunked_array.hpp"

namespace xt
{
    using chunked_array = xt::xchunked_array<xt::xarray<double>>;

    TEST(xchunked_array, indexed_access)
    {
        chunked_array a(
            {10, 10, 10},
            {2, 3, 4}
        );
        std::vector<size_t> idx = {3, 9, 8};

        a[idx] = 4.;
        ASSERT_EQ(a(3, 9, 8), 4.);

        a(3, 9, 8) = 5.;
        ASSERT_EQ(a(3, 9, 8), 5.);
    }
}
