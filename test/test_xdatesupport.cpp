/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#if defined(__GNUC__) && (__GNUC__ == 7) && (__cplusplus == 201703L)
#warning "test_xdatesupport.cpp has been deactivated because it leads to internal compiler error"
#else

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
    using tp = std::chrono::system_clock::time_point;

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

// need to wait until the system clock on Windows catches up with Linux
#ifndef _MSC_VER
    TEST(xdate, date_arange)
    {
        xarray<tp> tarr = xt::arange<tp>(std::chrono::system_clock::now(),
                                         std::chrono::system_clock::now() + std::chrono::hours(15),
                                         std::chrono::hours(1));
        EXPECT_TRUE(tarr.storage().back() > tarr.storage().front());
    }
#endif

    TEST(xdate, xfunction)
    {
        xarray<tp> tarr = { std::chrono::system_clock::now(),
                            std::chrono::system_clock::now(),
                            std::chrono::system_clock::now() };

        auto hours = std::chrono::hours(15);

        auto func = tarr + hours;
        xarray<tp> arrpf = func;

        arrpf(0) -= std::chrono::hours(200);

        EXPECT_TRUE(all(equal(tarr, tarr)));
        xarray<bool> cmp_res = {true, false, false};
        EXPECT_EQ(cmp_res, (arrpf < tarr));
        EXPECT_EQ(!cmp_res, (arrpf > tarr));
        cmp_res = {false, false, false};
        EXPECT_EQ(cmp_res, equal(tarr, arrpf));
    }
}

#endif // defined(__GNUC__) && (__GNUC__ == 7) && (__cplusplus == 201703L)

