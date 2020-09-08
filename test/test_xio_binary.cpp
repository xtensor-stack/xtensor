/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xio_binary.hpp"

namespace xt
{
    TEST(xio_binary, dump_load)
    {
        xtensor<double, 2> data
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        const char* fname = "data.bin";
        std::ofstream out_file(fname, std::ofstream::binary);
        dump_file(out_file, data, xio_binary_config());

        xarray<double> a;
        std::ifstream in_file(fname, std::ifstream::binary);
        load_file(in_file, a, xio_binary_config());
        a.reshape({2, 4});

        ASSERT_TRUE(all(equal(a, data)));
    }
}
