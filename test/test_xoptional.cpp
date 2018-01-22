/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "xtensor/xio.hpp"
#include "xtensor/xoptional.hpp"

namespace xt
{
    TEST(xoptional, tensor)
    {
        xtensor_optional<double, 2> m
            {{ 1.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        ASSERT_EQ(m(0, 0).value(), 1.0);
        ASSERT_EQ(m(1, 0).value(), 3.0);
        ASSERT_FALSE(m(1, 1).has_value());
        ASSERT_EQ((m[{0, 0}].value()), 1.0);
    }

    TEST(xoptional, operation)
    {
        xtensor_optional<double, 2> m1
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        xtensor<double, 2> m2
            {{ 1.0 , 2.0 },
             { 3.0 , 1.0 }};

        auto res_add = m1 + m2;
        ASSERT_EQ(res_add(0, 0).value(), 1.0);
        ASSERT_EQ(res_add(1, 0).value(), 6.0);
        ASSERT_FALSE(res_add(1, 1).has_value());
        ASSERT_EQ(res_add.value()(0, 0), 1.0);
        ASSERT_TRUE(res_add.has_value()(0, 0));
        ASSERT_FALSE(res_add.has_value()(1, 1));

        auto res_mul = m1 * m2;
        ASSERT_EQ(res_mul(0, 0).value(), 0.0);
        ASSERT_EQ(res_mul(1, 0).value(), 9.0);
        ASSERT_FALSE(res_mul(1, 1).has_value());

        auto res_div = m1 / m2;
        ASSERT_EQ(res_div(0, 0).value(), 0.0);
        ASSERT_EQ(res_div(1, 0).value(), 1.0);
        ASSERT_FALSE(res_div(1, 1).has_value());

        xtensor_optional<double, 2> res = m1 + m2;
        ASSERT_EQ(res(0, 0).value(), 1.0);
        ASSERT_EQ(res(1, 0).value(), 6.0);
        ASSERT_FALSE(res(1, 1).has_value());
    }

    TEST(xoptional, xio)
    {
        std::ostringstream oss;
        xtensor_optional<double, 2> m
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        oss << m;
        std::string expect = "{{  0,   2},\n {  3, N/A}}";
        ASSERT_EQ(oss.str(), expect);
    }

    TEST(xoptional, ufunc)
    {
        xtensor_optional<double, 2> m
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        auto flag_view = xt::has_value(m);

        xtensor<bool, 2> res = flag_view;

        ASSERT_TRUE(res(0, 0));
        ASSERT_TRUE(res(0, 1));
        ASSERT_TRUE(res(1, 0));
        ASSERT_FALSE(res(1, 1));

        auto value_view = xt::value(m);

        xtensor<double, 2> resv = value_view;
        flag_view(1, 1) = true;
        ASSERT_TRUE(m(1, 1).has_value());
        value_view(1, 1) = 4.0;
        ASSERT_EQ(m(1, 1).value(), 4.0);
    }

    TEST(xoptional, ufunc_nonoptional)
    {
        xtensor<double, 2> m
            {{ 0.0 , 2.0 },
             { 3.0 , 1.0 }};

        auto flag_view = has_value(m);

        xtensor<bool, 2> res = flag_view;
        ASSERT_TRUE(res(0, 0));
        ASSERT_TRUE(res(0, 1));
        ASSERT_TRUE(res(1, 0));
        ASSERT_TRUE(res(1, 1));
    }

    TEST(xoptional, compilation)
    {
        using functor_type = detail::plus<double>;
        using tensor_type = xtensor_optional<double, 2>;
        using function_type = xoptional_function<functor_type, xtl::xoptional<double>, tensor_type, tensor_type>;

        tensor_type t1, t2;
        function_type f(functor_type(), t1, t2);
    }

#define UNARY_OPTIONAL_TEST_IMPL(FUNC)                                         \
    xtensor_optional<double, 2> m1{{0.25, 1}, {0.75, xtl::missing<double>()}}; \
    xtensor<double, 2> m2{{0.25, 1}, {0.75, 1}};                               \
    ASSERT_TRUE(FUNC(m1)(0, 1).has_value());                                   \
    ASSERT_EQ(FUNC(m2)(0, 1), FUNC(m1)(0, 1).value());                         \
    ASSERT_FALSE(FUNC(m1)(1, 1).has_value());

#define UNARY_OPTIONAL_TEST(FUNC)      \
    TEST(xoptional, FUNC)              \
    {                                  \
        UNARY_OPTIONAL_TEST_IMPL(FUNC) \
    }

#define UNARY_OPTIONAL_TEST_QUALIFIED(FUNC) \
    TEST(xoptional, FUNC)                   \
    {                                       \
        UNARY_OPTIONAL_TEST_IMPL(xt::FUNC)  \
    }

#define BINARY_OPTIONAL_TEST(FUNC)                                                   \
    TEST(xoptional, FUNC)                                                            \
    {                                                                                \
        xtensor_optional<double, 2> m1{{0.25, 0.5}, {0.75, xtl::missing<double>()}}; \
        xtensor_optional<double, 2> m2{{0.25, xtl::missing<double>()}, {0.75, 1.}};  \
        xtensor<double, 2> m3{{0.25, 0.5}, {0.75, 1.}};                              \
        ASSERT_TRUE(FUNC(m1, m3)(0, 1).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(0, 1), FUNC(m1, m3)(0, 1).value());                   \
        ASSERT_FALSE(FUNC(m1, m3)(1, 1).has_value());                                \
        ASSERT_TRUE(FUNC(m3, m1)(0, 1).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(0, 1), FUNC(m3, m1)(0, 1).value());                   \
        ASSERT_FALSE(FUNC(m3, m1)(1, 1).has_value());                                \
        ASSERT_TRUE(FUNC(m1, m2)(1, 0).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(1, 0), FUNC(m1, m2)(1, 0).value());                   \
        ASSERT_FALSE(FUNC(m1, m2)(0, 1).has_value());                                \
        ASSERT_FALSE(FUNC(m1, m2)(1, 1).has_value());                                \
    }

#define TERNARY_OPTIONAL_TEST_IMPL(FUNC)                                         \
    xtensor_optional<double, 2> m1{{0.25, 0.5}, {0.75, xtl::missing<double>()}}; \
    xtensor<double, 2> m4{{0.25, 0.5}, {0.75, 1.}};                              \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m4, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m4, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m1, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m1, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m4, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m4, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m1, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m1, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m4, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m4, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m1, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m1, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m1, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m1, m1)(1, 1).has_value());

#define TERNARY_OPTIONAL_TEST(FUNC)          \
    TEST(xoptional, FUNC)                    \
    {                                        \
        TERNARY_OPTIONAL_TEST_IMPL(xt::FUNC) \
    }

    UNARY_OPTIONAL_TEST(abs)
    UNARY_OPTIONAL_TEST(fabs)
    BINARY_OPTIONAL_TEST(fmod)
    BINARY_OPTIONAL_TEST(remainder)
    TERNARY_OPTIONAL_TEST(fma)
    BINARY_OPTIONAL_TEST(fmax)
    BINARY_OPTIONAL_TEST(fmin)
    BINARY_OPTIONAL_TEST(fdim)
    UNARY_OPTIONAL_TEST(sign)
    UNARY_OPTIONAL_TEST(exp)
    UNARY_OPTIONAL_TEST(exp2)
    UNARY_OPTIONAL_TEST(expm1)
    UNARY_OPTIONAL_TEST(log)
    UNARY_OPTIONAL_TEST(log10)
    UNARY_OPTIONAL_TEST(log2)
    UNARY_OPTIONAL_TEST(log1p)
    BINARY_OPTIONAL_TEST(pow)
    UNARY_OPTIONAL_TEST(sqrt)
    UNARY_OPTIONAL_TEST(cbrt)
    BINARY_OPTIONAL_TEST(hypot)
    UNARY_OPTIONAL_TEST(sin)
    UNARY_OPTIONAL_TEST(cos)
    UNARY_OPTIONAL_TEST(tan)
    UNARY_OPTIONAL_TEST(acos)
    UNARY_OPTIONAL_TEST(asin)
    UNARY_OPTIONAL_TEST(atan)
    BINARY_OPTIONAL_TEST(atan2)
    UNARY_OPTIONAL_TEST(sinh)
    UNARY_OPTIONAL_TEST(cosh)
    UNARY_OPTIONAL_TEST(tanh)
    UNARY_OPTIONAL_TEST(acosh)
    UNARY_OPTIONAL_TEST(asinh)
    UNARY_OPTIONAL_TEST(atanh)
    UNARY_OPTIONAL_TEST(erf)
    UNARY_OPTIONAL_TEST(erfc)
    UNARY_OPTIONAL_TEST(tgamma)
    UNARY_OPTIONAL_TEST(lgamma)
    UNARY_OPTIONAL_TEST_QUALIFIED(isfinite)
    UNARY_OPTIONAL_TEST_QUALIFIED(isinf)
    UNARY_OPTIONAL_TEST_QUALIFIED(isnan)

#undef TERNARY_OPTIONAL_TEST
#undef TERNARY_OPTIONAL_TEST_IMPL
#undef BINARY_OPTIONAL_TEST
#undef UNARY_OPTIONAL_TEST_QUALIFIED
#undef UNARY_OPTIONAL_TEST
#undef UNARY_OPTIONAL_TEST_IMPL
}
