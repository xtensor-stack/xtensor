#include "gtest/gtest.h"

#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>

#include "xarray/xarray.hpp"
#include "xarray/xio.hpp"


namespace qs
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
        std::cout << e;
        std::stringstream out;
        out << e;
        ASSERT_EQ(out.str(), "{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}");
    }

}

