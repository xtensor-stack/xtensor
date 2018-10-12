/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <chrono>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    using days            = std::chrono::duration<long long, std::ratio<3600 * 24>>;
    using days_time_point = std::chrono::time_point<std::chrono::system_clock, days>;

    std::ostream& operator<<(std::ostream& os, const days_time_point& /*rhs*/)
    {
        // Too many problems with puttime on old compilers, so removing it.
        // std::time_t x = std::chrono::system_clock::to_time_t(rhs);
        // os << std::put_time(std::gmtime(&x), "%F %T");
        os << "puttime here";
        return os;
    }

    TEST(xdate, xarray_of_dates)
    {
        xt::xarray<days_time_point> dates(
            {
                days_time_point{days{300}},
                days_time_point{days{400}},
                days_time_point{days{600}},
                days_time_point{days{10000}}
            });

        xt::xarray<days> durations({days{300}, days{400}, days{600}, days{10000}});
        
        xt::xarray<days_time_point> result = dates + durations;
        xt::xarray<days_time_point> result2 = dates + days{500};

        xt::xarray<bool> expected = {true, true, false, false};

        EXPECT_EQ((result < result2), expected);
    }
}
