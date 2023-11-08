/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <sstream>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xinfo.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(xinfo, compiles)
    {
        xarray<double> test = {{1, 2, 3}, {4, 5, 6}};
        std::stringstream ss;

        ss << info(test) << std::endl;
    }

    // See https://github.com/xtensor-stack/xtensor/issues/2694
    TEST(xinfo, typename)
    {
        xarray<double> test = {{1, 2, 3}, {4, 5, 6}};
        auto t_s = type_to_string<typename decltype(test)::value_type>();
        std::string expected = "double";
#if defined(__clang__)
        WARN_EQ(expected, t_s);
#else
        EXPECT_EQ(expected, t_s);
#endif
    }
}
