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

#include "xarray/xarray.hpp"
#include "xarray/xio.hpp"


namespace xt
{

    TEST(xio, simple)
    {
        xshape<size_t> shape = {3, 4};
        xarray<double> e(shape);
        std::vector<double> data {
        	1, 2, 3, 4,
        	5, 6, 7, 8,
        	9, 10, 11, 12
        };
        std::copy(data.begin(), data.end(), e.storage_begin());
        std::stringstream out;
        out << e;
        ASSERT_EQ(out.str(), "{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}");
    }

}

