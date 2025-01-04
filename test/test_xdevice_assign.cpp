/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
// This file is generated from test/files/cppy_source/test_extended_broadcast_view.cppy by preprocess.py!
// Warning: This file should not be modified directly! Instead, modify the `*.cppy` file.


#include <algorithm>

#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(test_xdevice, basic_xfunction)
    {
        std::vector<double> expectation = {2,3,4,5,6};

        xt::xarray<float> a = {1., 2., 3., 4., 5.};
        xt::xarray<float> b = xt::ones_like(a);
        auto c = xt::xtensor<float, 1>::from_shape(a.shape());
	    c = a + b;
        for(size_t i = 0; i < expectation.size(); i++)
        {
            ASSERT_EQ(c(i), expectation.at(i));
        }
    }
}
