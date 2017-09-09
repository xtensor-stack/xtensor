/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xmathutil.hpp"

#include <limits>

namespace xt
{
    template <class T>
    auto test_abs(T t)
    {
        using namespace cmath;
        return abs(t);
    }

    TEST(xmathutil, cmath)
    {
        EXPECT_TRUE((std::is_same<decltype(test_abs(1)), int>::value));

        // without XTENSOR_DEFINE_UNSIGNED_ABS, this causes 'ambiguous call to overloaded function'
        EXPECT_TRUE((std::is_same<decltype(test_abs(1u)), unsigned int>::value));

        EXPECT_TRUE((std::is_same<decltype(cmath::floor(1.1)), double>::value));

        // without XTENSOR_DEFINE_INTEGER_FLOOR_CEIL, this results in 'double'
        EXPECT_TRUE((std::is_same<decltype(cmath::floor(1)), int>::value));
    }

    TEST(xmathutil, isclose)
    {
        double eps = std::numeric_limits<double>::epsilon();

        // test default tolerance
        EXPECT_TRUE(isclose(numeric_constants<>::PI, 3.141592653));
        EXPECT_FALSE(isclose(numeric_constants<>::PI, 3.141));
        // test custom tolerance
        EXPECT_TRUE(isclose(numeric_constants<>::PI, 3.141592653589793238463, eps, eps));
        EXPECT_FALSE(isclose(numeric_constants<>::PI, 3.141592653, eps, eps));
        EXPECT_TRUE(isclose(numeric_constants<>::PI, 3.141, 1e-3));
        EXPECT_FALSE(isclose(numeric_constants<>::PI, 3.141, 1e-4));
        EXPECT_TRUE(isclose(numeric_constants<>::PI, 3.141, 1e-4, 1e-3));
        // test NaN
        EXPECT_FALSE(isclose(std::log(-1.0), 3.141));
        EXPECT_FALSE(isclose(std::log(-1.0), std::log(-2.0)));
        EXPECT_TRUE(isclose(std::log(-1.0), std::log(-2.0), eps, eps, true));
    }

    TEST(xmathutil, functions)
    {
        EXPECT_EQ(sq(2), 4);
        EXPECT_EQ(sq(1.5), 2.25);
        EXPECT_EQ(dot(2, 3), 6);
        EXPECT_EQ(dot(1.5, 2.5), 3.75);

        EXPECT_EQ(min(1.5, 2), 1.5);
        EXPECT_EQ(max(1.5, 2), 2.0);

        EXPECT_TRUE(even(2));
        EXPECT_FALSE(even(3));
        EXPECT_TRUE(odd(3));
        EXPECT_FALSE(odd(2));

        EXPECT_EQ(sin_pi(1.0), 0.0);
        EXPECT_EQ(sin_pi(1.5), -1.0);
        EXPECT_EQ(cos_pi(1.0), -1.0);
        EXPECT_EQ(cos_pi(1.5), 0.0);

        EXPECT_EQ(norm(2), 2);
        EXPECT_EQ(norm(-2), 2);
        EXPECT_EQ(norm(2.0), 2.0);
        EXPECT_EQ(norm(-2.0), 2.0);

        EXPECT_EQ(squared_norm(2), 4);
        EXPECT_EQ(squared_norm(-2), 4);
        EXPECT_EQ(squared_norm(2.0), 4.0);
        EXPECT_EQ(squared_norm(-2.0), 4.0);
    }
} // namespace xt