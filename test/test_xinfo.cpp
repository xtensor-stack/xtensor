/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <sstream>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xinfo.hpp"

namespace xt
{
    TEST(xinfo, compiles)
    {
    	xarray<double> test = {{1,2,3}, {4,5,6}};
    	std::stringstream ss;

    	ss << info(test) << std::endl;
    }

    TEST(xinfo, typename)
    {
    	xarray<double> test = {{1,2,3}, {4,5,6}};
    	auto t_s = type_to_string<typename decltype(test)::value_type>();
    	std::string expected = "double";
    	EXPECT_EQ(expected, t_s);
    }
}