/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>

#include "xtensor/xmissing.hpp"

namespace xt
{
    TEST(xoptional, scalar_tests)
    {
        // Test uninitialized == missing
        xoptional<double, bool> v0;
        ASSERT_FALSE(v0.has_value());

        // Test initialization from value
        xoptional<double, bool> v1(1.0);
        ASSERT_TRUE(v1.has_value());
        ASSERT_EQ(v1.value(), 1.0);

        // Test lvalue closure types
        double value1 = 3.0;
        int there = 0;
        auto opt1 = optional(value1, there);
        ASSERT_FALSE(opt1.has_value());
        opt1 = 1.0;
        ASSERT_TRUE(opt1.has_value());
        ASSERT_EQ(value1, 1.0);

        // Test rvalue closure type for boolean
        double value2 = 3.0;
        auto opt2 = optional(value2, true);
        opt2 = 2.0;
        ASSERT_TRUE(opt2.has_value());
        ASSERT_EQ(value2, 2.0);
    }

    TEST(xoptional, vector)
    {
        xoptional_vector<double> v(3, 2.0);
        ASSERT_TRUE(v.front().has_value());
        ASSERT_TRUE(v[0].has_value());
        ASSERT_EQ(v[0].value(), 2.0);
        v[1] = missing<double>();
        ASSERT_FALSE(v[1].has_value());
    }

    TEST(xoptional, vector_iteration)
    {
        xoptional_vector<double> v(4, 2.0);
        v[0] = missing<double>();
        std::vector<double> res;
        for(auto it = v.cbegin(); it != v.cend(); ++it)
        {
            res.push_back((*it).value_or(0.0));
        }
        std::vector<double> expect = {0.0, 2.0, 2.0, 2.0};
        ASSERT_TRUE(std::equal(res.begin(), res.end(), expect.begin()));
    }

    TEST(xoptional, tensor)
    {
        xtensor_optional<double, 2> m

            {{ 1.0 ,       2.0          },
             { 3.0 , missing<double>()} };

        ASSERT_EQ(m(0, 0).value(), 1.0);
        ASSERT_EQ(m(1, 0).value(), 3.0);
        ASSERT_FALSE(m(1, 1).has_value());
    }
}

