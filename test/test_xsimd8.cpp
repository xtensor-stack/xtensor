/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <complex>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    using namespace std::complex_literals;

    TEST(xcomplex, arg)
    {
        xarray<std::complex<double>> cmplarg_0 = {
            {0.40101756 + 0.71233018i, 0.62731701 + 0.42786349i, 0.32415089 + 0.2977805i},
            {0.24475928 + 0.49208478i, 0.69475518 + 0.74029639i, 0.59390240 + 0.35772892i},
            {0.63179202 + 0.41720995i, 0.44025718 + 0.65472131i, 0.08372648 + 0.37380143i}};
        xarray<std::complex<double>> res = xt::conj(cmplarg_0);
    }
}
