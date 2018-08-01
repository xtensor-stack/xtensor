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

    TEST(xmasked_value, ctor)
    {
        double a = 5.2;

        auto m = xmasked_value<double&>(a);
        EXPECT_EQ(m.value(), 5.2);
        EXPECT_EQ(m.visible(), true);
    }

    TEST(xmasked_value, value)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<optional_type>(b);

        EXPECT_EQ(c.value(), 5.2);
        a = 3.35;
        EXPECT_EQ(c.value(), 3.35);
        c.value() = 126.;
        EXPECT_EQ(c.value(), 126.);
        EXPECT_EQ(a, 126.);

        auto c2 = xmasked_value<double>(3.);
        EXPECT_EQ(c2.value(), 3.);
        c2.value() = 36.;
        EXPECT_EQ(c2.value(), 36.);
    }

    TEST(xmasked_value, visible)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<optional_type, bool&>(b, bool_val);

        EXPECT_TRUE(c.visible());
        bool_val = false;
        EXPECT_FALSE(c.visible());
    }

    TEST(xmasked_value, conversion)
    {
        double a = 5.2;
        bool bool_val = true;

        auto b = xmasked_value<double&, bool&>(a, bool_val);

        xtl::xoptional<double&, bool> c = b;
        EXPECT_EQ(c.value(), 5.2);
        EXPECT_TRUE(c.has_value());

        double& val = b;
        EXPECT_EQ(val, 5.2);
        val = 36.;
        EXPECT_EQ(c.value(), 36.);
        EXPECT_EQ(b.value(), 36.);
    }

    TEST(xmasked_value, comparison)
    {
        double a = 5.2;
        bool bool_val = true;

        auto m1 = xmasked_value<double&, bool&>(a, bool_val);
        auto m2 = xmasked_value<double&, bool&>(a, bool_val);

        EXPECT_EQ(m1, m2);
        EXPECT_EQ(m2, m1);

        EXPECT_EQ(m1, 5.2);
        EXPECT_EQ(5.2, m1);

        EXPECT_NE(m1, 6.2);
        EXPECT_NE(6.2, m1);

        bool_val = false;

        EXPECT_NE(m1, 5.2);
        EXPECT_NE(5.2, m1);

        auto masked1 = masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(m1, masked1);
        EXPECT_EQ(m2, masked1);

        auto o1 = xtl::xoptional<double, bool>(3.5, true);
        auto o2 = xtl::xoptional<double, bool>(6.5, false);
        auto m3 = xmasked_value<xtl::xoptional<double, bool>, bool>(o1, true);
        auto m4 = xmasked_value<xtl::xoptional<double, bool>, bool>(o2, true);

        EXPECT_EQ(m3, o1);
        EXPECT_EQ(o1, m3);
        EXPECT_NE(m4, o1);
        EXPECT_NE(o2, m3);
        EXPECT_NE(m3, m4);

        m3.visible() = m4.visible() = false;
        EXPECT_EQ(m3, m4);
        auto masked2 = masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(m3, masked2);
        EXPECT_EQ(m4, masked2);
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

        auto o1 = xtl::xoptional<double, bool>(3.5, true);
        auto o2 = xtl::xoptional<double, bool>(6.5, false);
        auto m3 = xmasked_value<xtl::xoptional<double, bool>, bool>(o1, true);
        auto m4 = xmasked_value<xtl::xoptional<double, bool>, bool>(o2, false);

        swap(m3, m4);

        EXPECT_EQ(m3.value().value(), 6.5);
        EXPECT_FALSE(m3.value().has_value());
        EXPECT_FALSE(m3.visible());
        EXPECT_EQ(m4.value().value(), 3.5);
        EXPECT_TRUE(m4.value().has_value());
        EXPECT_TRUE(m4.visible());
    }

    TEST(xmasked_value, arithm_neg)
    {
        double a = 5.2;
        bool b = true;

        auto m = xmasked_value<double&, bool&>(a, b);

        xmasked_value<double, bool> r = -m;
        EXPECT_EQ(r.value(), -5.2);
        EXPECT_EQ(r.visible(), true);
    }

    TEST(xmasked_value, arithm_plus)
    {
        double a = 5.2;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<optional_type>(o);

        auto r1 = m + o;
        EXPECT_EQ(r1.value(), 10.4);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o + m;
        EXPECT_EQ(r2.value(), 10.4);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m + m;
        EXPECT_EQ(r3.value(), 10.4);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 4.2 + m;
        EXPECT_EQ(r4.value(), 9.4);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m + 4.2;
        EXPECT_EQ(r5.value(), 9.4);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, arithm_minus)
    {
        double a = 5.2;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<optional_type>(o);

        auto r1 = m - o;
        EXPECT_EQ(r1.value(), 0.0);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o - m;
        EXPECT_EQ(r2.value(), 0.0);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m - m;
        EXPECT_EQ(r3.value(), 0.0);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 4.2 - m;
        EXPECT_EQ(r4.value(), -1.0);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m - 4.2;
        EXPECT_EQ(r5.value(), 1.0);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, arithm_mult)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<optional_type>(o);

        auto r1 = m * o;
        EXPECT_EQ(r1.value(), 25.);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o * m;
        EXPECT_EQ(r2.value(), 25.);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m * m;
        EXPECT_EQ(r3.value(), 25.);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 4. * m;
        EXPECT_EQ(r4.value(), 20.);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m * 4.;
        EXPECT_EQ(r5.value(), 20.);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, arithm_div)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<optional_type>(o);

        auto r1 = m / o;
        EXPECT_EQ(r1.value(), 1.);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o / m;
        EXPECT_EQ(r2.value(), 1.);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m / m;
        EXPECT_EQ(r3.value(), 1.);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 25. / m;
        EXPECT_EQ(r4.value(), 5.);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m / 1.;
        EXPECT_EQ(r5.value(), 5.);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, operator_or)
    {
        bool b = true;

        auto o = xtl::xoptional<bool, bool&>(true, b);
        auto m = xmasked_value<xtl::xoptional<bool, bool&>, bool>(o, true);

        auto r1 = m || o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o || m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m || m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = true || m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = false || m;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.visible(), true);

        auto r6 = m || true;
        EXPECT_EQ(r6.value(), true);
        EXPECT_EQ(r6.visible(), true);

        auto r7 = m || false;
        EXPECT_EQ(r7.value(), true);
        EXPECT_EQ(r7.visible(), true);
    }

    TEST(xmasked_value, operator_and)
    {
        bool b = true;

        auto o = xtl::xoptional<bool, bool&>(true, b);
        auto m = xmasked_value<xtl::xoptional<bool, bool&>, bool>(o, true);

        auto r1 = m && o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o && m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m && m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = true && m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = false && m;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.visible(), true);

        auto r6 = m && true;
        EXPECT_EQ(r6.value(), true);
        EXPECT_EQ(r6.visible(), true);

        auto r7 = m && false;
        EXPECT_EQ(r7.value(), false);
        EXPECT_EQ(r7.visible(), true);

        m.visible() = false;
        auto r8 = m && true;
        EXPECT_EQ(r8.visible(), false);
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
        auto o2 = optional_type(a2, b);

        auto m = xmasked_value<optional_type, bool>(o2, true);

        auto r1 = m < o;
        EXPECT_EQ(r1.value(), false);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o < m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m < m;
        EXPECT_EQ(r3.value(), false);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 25. < m;
        EXPECT_EQ(r4.value(), false);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m < 25.;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, operator_less_equal)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);
        auto o2 = optional_type(a2, b);

        auto m = xmasked_value<optional_type, bool>(o2, true);

        auto r1 = m <= o;
        EXPECT_EQ(r1.value(), false);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o <= m;
        EXPECT_EQ(r2.value(), true);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m <= m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 25. <= m;
        EXPECT_EQ(r4.value(), false);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m <= 25.;
        EXPECT_EQ(r5.value(), true);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, operator_more)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);
        auto o2 = optional_type(a2, b);

        auto m = xmasked_value<optional_type, bool&>(o2, b);

        auto r1 = m > o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o > m;
        EXPECT_EQ(r2.value(), false);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m > m;
        EXPECT_EQ(r3.value(), false);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 25. > m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m > 25.;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, operator_more_equal)
    {
        double a1 = 5.0;
        double a2 = 10.0;
        bool b = true;

        auto o = optional_type(a1, b);
        auto o2 = optional_type(a2, b);

        auto m = xmasked_value<optional_type, bool&>(o2, b);

        auto r1 = m >= o;
        EXPECT_EQ(r1.value(), true);
        EXPECT_EQ(r1.visible(), true);

        auto r2 = o >= m;
        EXPECT_EQ(r2.value(), false);
        EXPECT_EQ(r2.visible(), true);

        auto r3 = m >= m;
        EXPECT_EQ(r3.value(), true);
        EXPECT_EQ(r3.visible(), true);

        auto r4 = 25. >= m;
        EXPECT_EQ(r4.value(), true);
        EXPECT_EQ(r4.visible(), true);

        auto r5 = m >= 25.;
        EXPECT_EQ(r5.value(), false);
        EXPECT_EQ(r5.visible(), true);
    }

    TEST(xmasked_value, assign)
    {
        double a = 5.2;
        double a2 = 3.35;
        bool bool_val = true;

        auto b = optional_type(a, bool_val);
        auto c = xmasked_value<optional_type>(b);

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
        EXPECT_NEAR(a, 6.7, 0.0001);
        c -= b;
        EXPECT_NEAR(a, 0., 0.0001);
        c = 3.35;

        // When masked, the assign doesn't do anything
        c.visible() = false;
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

        auto o = xtl::xoptional<double&>(a1);
        auto m1 = xmasked_value<xtl::xoptional<double&>, bool&>(o, bool_val);
        auto m2 = xmasked_value<double&, bool&>(a2, bool_val);

        EXPECT_EQ(xt::pow(m1, m1).value(), std::pow(5., 5.));
        EXPECT_EQ(xt::pow(m1, o).value(), std::pow(5., 5.));
        EXPECT_EQ(xt::pow(o, m1).value(), std::pow(5., 5.));
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
        auto masked1 = masked<double>();

        EXPECT_EQ(xt::fma(m1, m2, m2).value(), 66.);
        EXPECT_EQ(xt::fma(m1, masked1, m2), masked1);

        EXPECT_EQ(xt::fma(m1, m2, 2.).value(), 57.);
        EXPECT_EQ(xt::fma(m1, 2., m2).value(), 21.);
        EXPECT_EQ(xt::fma(2., m1, m2).value(), 21.);
        EXPECT_EQ(xt::fma(2., 10., m2).value(), 31.);
        EXPECT_EQ(xt::fma(2., m2, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(m2, 5., 3.).value(), 58.);
        EXPECT_EQ(xt::fma(m2, 5., masked1), masked1);

        auto o = xtl::xoptional<double>(2.);
        auto o1 = xtl::xoptional<double>(5.);
        auto o2 = xtl::xoptional<double>(11.);
        auto m3 = xmasked_value<xtl::xoptional<double>, bool&>(o1, bool_val);
        auto m4 = xmasked_value<xtl::xoptional<double>, bool&>(o2, bool_val);
        auto masked2 = masked<xtl::xoptional<double>>();

        EXPECT_EQ(xt::fma(m3, m4, o).value(), 57.);
        EXPECT_EQ(xt::fma(m3, o, m4).value(), 21.);
        EXPECT_EQ(xt::fma(o, m3, m4).value(), 21.);
        EXPECT_EQ(xt::fma(o, m3, masked2), masked2);
        EXPECT_EQ(xt::fma(o, o, m4).value(), 15.);
        EXPECT_EQ(xt::fma(o, m4, o).value(), 24.);
        EXPECT_EQ(xt::fma(m4, o, o).value(), 24.);

        EXPECT_EQ(xt::fma(3., m4, o).value(), 35.);
        EXPECT_EQ(xt::fma(m4, 3., o).value(), 35.);
        EXPECT_EQ(xt::fma(m4, o, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(o, m4, 3.).value(), 25.);
        EXPECT_EQ(xt::fma(o, 3., m3).value(), 11.);
        EXPECT_EQ(xt::fma(3., o, m3).value(), 11.);
    }
}
