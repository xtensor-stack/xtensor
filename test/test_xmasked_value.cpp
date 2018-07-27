/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay Wolf Vollprecht and    *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xmasked_value.hpp"

namespace xt
{
    using optional_type = xtl::xoptional<double&, bool&>;

    TEST(xmasked_value, value)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<double&, bool&>(b);

        EXPECT_EQ(c.value(), 5.2);
        a = 3.35;
        EXPECT_EQ(c.value(), 3.35);
        c.value() = 126.;
        EXPECT_EQ(c.value(), 126.);
        EXPECT_EQ(a, 126.);
    }

    TEST(xmasked_value, has_value)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<double&, bool&>(b);

        EXPECT_TRUE(c.has_value());
        bool_val = false;
        EXPECT_FALSE(c.has_value());
    }

    TEST(xmasked_value, conversion_to_optional)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = xmasked_value<double&, bool&>(a, bool_val);

        optional_type c = b;
        EXPECT_EQ(c.value(), 5.2);
        EXPECT_TRUE(c.has_value());
    }

    TEST(xmasked_value, comparison)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = xmasked_value<double&, bool&>(a, bool_val);
        auto c = xmasked_value<double&, bool&>(a, bool_val);

        EXPECT_EQ(b, c);
        EXPECT_EQ(c, b);

        EXPECT_EQ(b, optional_type(a, bool_val));
        EXPECT_EQ(optional_type(a, bool_val), b);

        EXPECT_EQ(b, 5.2);
        EXPECT_EQ(5.2, b);

        EXPECT_NE(b, 6.2);
        EXPECT_NE(6.2, b);

        bool_val = false;

        EXPECT_NE(b, 5.2);
        EXPECT_NE(5.2, b);

        EXPECT_EQ(b, optional_type(a, bool_val));
        EXPECT_EQ(optional_type(a, bool_val), b);
    }

    TEST(xmasked_value, swap)
    {
        double a1 = 5.2;
        bool b1 = true;
        double a2 = 36.5;
        bool b2 = false;

        auto m1 = xmasked_value<double&, bool&>(a1, b1);
        auto m2 = xmasked_value<double&, bool&>(a2, b2);

        EXPECT_EQ(a1, 5.2);
        EXPECT_EQ(b1, true);
        EXPECT_EQ(a2, 36.5);
        EXPECT_EQ(b2, false);

        m1.swap(m2);

        EXPECT_EQ(a1, 36.5);
        EXPECT_EQ(b1, false);
        EXPECT_EQ(a2, 5.2);
        EXPECT_EQ(b2, true);
    }

    TEST(xmasked_value, arithm_neg)
    {
        double a = 5.2;
        bool b = true;

        auto m = xmasked_value<double&, bool&>(a, b);

        xmasked_value<double, bool> r = -m;
        EXPECT_EQ(r.value(), -5.2);
        EXPECT_EQ(r.has_value(), true);
    }

    TEST(xmasked_value, arithm_plus)
    {
        double a = 5.2;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        auto r1 = m + o;
        EXPECT_EQ(r1.value(), 10.4);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o + m;
        EXPECT_EQ(r2.value(), 10.4);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m + m;
        EXPECT_EQ(r3.value(), 10.4);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 4.2 + m;
        EXPECT_EQ(r4.value(), 9.4);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m + 4.2;
        EXPECT_EQ(r5.value(), 9.4);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_minus)
    {
        double a = 5.2;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        auto r1 = m - o;
        EXPECT_EQ(r1.value(), 0.0);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o - m;
        EXPECT_EQ(r2.value(), 0.0);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m - m;
        EXPECT_EQ(r3.value(), 0.0);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 4.2 - m;
        EXPECT_EQ(r4.value(), -1.0);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m - 4.2;
        EXPECT_EQ(r5.value(), 1.0);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_mult)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        auto r1 = m * o;
        EXPECT_EQ(r1.value(), 25.);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o * m;
        EXPECT_EQ(r2.value(), 25.);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m * m;
        EXPECT_EQ(r3.value(), 25.);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 4. * m;
        EXPECT_EQ(r4.value(), 20.);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m * 4.;
        EXPECT_EQ(r5.value(), 20.);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_div)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        auto r1 = m / o;
        EXPECT_EQ(r1.value(), 1.);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o / m;
        EXPECT_EQ(r2.value(), 1.);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m / m;
        EXPECT_EQ(r3.value(), 1.);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 25. / m;
        EXPECT_EQ(r4.value(), 5.);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m / 1.;
        EXPECT_EQ(r5.value(), 5.);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, operator_or)
    {
        bool b = true;

        auto o = xtl::xoptional<bool, bool&>(true, b);
        auto m = xmasked_value<bool, bool&>(true, b);

        auto r1 = m || o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o || m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m || m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = true || m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = false || m;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.has_value(), true);

        auto r6 = m || true;
        EXPECT_EQ(r6.value(), true);
        EXPECT_EQ(r6.has_value(), true);

        auto r7 = m || false;
        EXPECT_EQ(r7.value(), true);
        EXPECT_EQ(r7.has_value(), true);
    }

    TEST(xmasked_value, operator_and)
    {
        bool b = true;

        auto o = xtl::xoptional<bool, bool&>(true, b);
        auto m = xmasked_value<bool, bool&>(true, b);

        auto r1 = m && o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o && m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m && m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = true && m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = false && m;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.has_value(), true);

        auto r6 = m && true;
        EXPECT_EQ(r6.value(), true);
        EXPECT_EQ(r6.has_value(), true);

        auto r7 = m && false;
        EXPECT_EQ(r7.value(), false);
        EXPECT_EQ(r7.has_value(), true);
    }

    TEST(xmasked_value, operator_not)
    {
        bool b = true;

        auto m1 = xmasked_value<bool, bool&>(true, b);
        auto m2 = xmasked_value<bool, bool&>(false, b);

        EXPECT_EQ(!m1, false);
        EXPECT_EQ(!m2, true);
    }

    TEST(xmasked_value, operator_less)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);

        auto m = xmasked_value<double&, bool&>(a2, b);

        auto r1 = m < o;
        EXPECT_EQ(r1.value(), false);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o < m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m < m;
        EXPECT_EQ(r3.value(), false);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 25. < m;
        EXPECT_EQ(r4.value(), false);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m < 25.;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, operator_less_equal)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);

        auto m = xmasked_value<double&, bool&>(a2, b);

        auto r1 = m <= o;
        EXPECT_EQ(r1.value(), false);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o <= m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m <= m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 25. <= m;
        EXPECT_EQ(r4.value(), false);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m <= 25.;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, operator_more)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);

        auto m = xmasked_value<double&, bool&>(a2, b);

        auto r1 = m > o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o > m;
        EXPECT_EQ(r2.value(), false);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m > m;
        EXPECT_EQ(r3.value(), false);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 25. > m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m > 25.;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, operator_more_equal)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);

        auto m = xmasked_value<double&, bool&>(a2, b);

        auto r1 = m >= o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.has_value(), true);

        auto r2 = o >= m;
        EXPECT_EQ(r2.value(), false);
        EXPECT_EQ(r2.has_value(), true);

        auto r3 = m >= m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.has_value(), true);

        auto r4 = 25. >= m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.has_value(), true);

        auto r5 = m >= 25.;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, assign)
    {
        double a = 5.2;
        double a2 = 3.35;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<double&, bool&>(b);

        // Value not masked, assigning a value works
        EXPECT_EQ(c.value(), 5.2);
        c = a2;
        EXPECT_EQ(c.value(), 3.35);
        EXPECT_EQ(a, 3.35);
        c += 1.;
        EXPECT_EQ(c.value(), 4.35);
        EXPECT_EQ(a, 4.35);
        c -= 1.;
        c *= 2.;
        EXPECT_NEAR(c.value(), 6.7, 0.0001);
        EXPECT_NEAR(a, 6.7, 0.0001);
        c = 3.35;

        // When masked, the assign doesn't do anything
        bool_val = false;
        c = 126.;
        EXPECT_EQ(c.value(), 3.35);
        EXPECT_EQ(a, 3.35);
        c *= 126;
        EXPECT_EQ(c.value(), 3.35);
        EXPECT_EQ(a, 3.35);
    }

    TEST(xmasked_value, unary_op)
    {
        double a1 = -5.2;
        double a2 = 5.2;
        bool bool_val = true;

        auto m1 = xmasked_value<double&, bool&>(a1, bool_val);
        auto m2 = xmasked_value<double&, bool&>(a2, bool_val);

        EXPECT_EQ(xt::abs(m1).value(), 5.2);
        EXPECT_EQ(xt::abs(m2).value(), 5.2);
    }

    TEST(xmasked_value, unary_bool_op)
    {
        double a = -5.2;
        bool bool_val = true;

        auto m = xmasked_value<double&, bool&>(a, bool_val);

        EXPECT_EQ(xt::isfinite(m).value(), true);
    }

    TEST(xmasked_value, binary_op)
    {
        double a1 = 5.0;
        double a2 = 5.0;
        bool bool_val = true;

        auto m1 = xmasked_value<double&, bool&>(a1, bool_val);
        auto m2 = xmasked_value<double&, bool&>(a2, bool_val);
        auto o = xtl::xoptional<double>(1.);

        EXPECT_EQ(xt::pow(m1, m2).value(), std::pow(5., 5.));
        EXPECT_EQ(xt::pow(m1, o).value(), 5.0);
        EXPECT_EQ(xt::pow(o, m1).value(), 1.0);
        EXPECT_EQ(xt::pow(m1, 2).value(), 25.0);
        EXPECT_EQ(xt::pow(2, m1).value(), 32);
    }

    TEST(xmasked_value, ternary_op)
    {
        double a1 = 5.0;
        double a2 = 11.0;
        bool bool_val = true;

        auto m1 = xmasked_value<double&, bool&>(a1, bool_val);
        auto m2 = xmasked_value<double&, bool&>(a2, bool_val);
        auto o = xtl::xoptional<double>(2.);

        EXPECT_EQ(xt::fma(m1, m2, m2).value(), 66.);
        EXPECT_EQ(xt::fma(m1, xtl::missing<double>(), m2), xtl::missing<double>());

        EXPECT_EQ(xt::fma(m1, m2, 2.).value(), 57.);
        EXPECT_EQ(xt::fma(m1, 2., m2).value(), 21.);
        EXPECT_EQ(xt::fma(2., m1, m2).value(), 21.);
        EXPECT_EQ(xt::fma(2., 10., m2).value(), 31.);
        EXPECT_EQ(xt::fma(2., m2, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(m2, 5., 3.).value(), 58.);
        EXPECT_EQ(xt::fma(m2, 5., xtl::missing<double>()), xtl::missing<double>());

        EXPECT_EQ(xt::fma(m1, m2, o).value(), 57.);
        EXPECT_EQ(xt::fma(m1, o, m2).value(), 21.);
        EXPECT_EQ(xt::fma(o, m1, m2).value(), 21.);
        EXPECT_EQ(xt::fma(o, o, m2).value(), 15.);
        EXPECT_EQ(xt::fma(o, m2, o).value(), 24.);
        EXPECT_EQ(xt::fma(m2, o, o).value(), 24.);

        EXPECT_EQ(xt::fma(3., m2, o).value(), 35.);
        EXPECT_EQ(xt::fma(m2, 3., o).value(), 35.);
        EXPECT_EQ(xt::fma(m2, o, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(o, m2, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(o, m2, xtl::missing<double>()), xtl::missing<double>());
        EXPECT_EQ(xt::fma(o, 3., m1).value(), 11.);
        EXPECT_EQ(xt::fma(3., o, m1).value(), 11.);
    }
}
