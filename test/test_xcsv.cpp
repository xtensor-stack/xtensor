/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <sstream>
#include <iostream>

#include "xtensor/xcsv.hpp"
#include "xtensor/xmath.hpp" 
#include "xtensor/xio.hpp" 

namespace xt
{
    TEST(xcsv, load_double)
    {
        std::string source =
            "1.0, 2.0, 3.0, 4.0\n"
            "10.0, 12.0, 15.0, 18.0";

        std::stringstream source_stream(source);

        xtensor<double, 2> res = load_csv<double>(source_stream);

        xtensor<double, 2> exp
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, dump_double)
    {
        xtensor<double, 2> data
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        std::stringstream res;

        dump_csv(res, data);
        ASSERT_EQ("1,2,3,4\n10,12,15,18\n", res.str());
    }
}
