/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay Wolf Vollprecht and    *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xio.hpp"

namespace xt
{
    using optional_type = xtl::xoptional<double&, bool&>;
    using data_type = xoptional_assembly<xarray<double>, xarray<bool>>;

    // data = {{ 1. ,  2., N/A },
    //         { N/A,  5.,  6. },
    //         { 7. ,  8.,  9. }}
    inline data_type make_test_data()
    {
        data_type d = {{ 1., 2., 3.},
                       { 4., 5., 6.},
                       { 7., 8., 9.}};
        d(0, 2).has_value() = false;
        d(1, 0).has_value() = false;
        return d;
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    inline auto make_masked_data(data_type& data)
    {
        xarray<bool> mask = {{ true,  true,  true},
                             {false, false, false},
                             { true,  true,  true}};

        return masked_view(data, std::move(mask));
    }

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

        xtl::xoptional<double, bool> r1 = m + o;
        EXPECT_EQ(r1.value(), 10.4);
        EXPECT_EQ(r1.has_value(), true);

        xtl::xoptional<double, bool> r2 = o + m;
        EXPECT_EQ(r2.value(), 10.4);
        EXPECT_EQ(r2.has_value(), true);

        xtl::xoptional<double, bool> r3 = m + m;
        EXPECT_EQ(r3.value(), 10.4);
        EXPECT_EQ(r3.has_value(), true);

        xtl::xoptional<double, bool> r4 = 4.2 + m;
        EXPECT_EQ(r4.value(), 9.4);
        EXPECT_EQ(r4.has_value(), true);

        xtl::xoptional<double, bool> r5 = m + 4.2;
        EXPECT_EQ(r5.value(), 9.4);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_minus)
    {
        double a = 5.2;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        xtl::xoptional<double, bool> r1 = m - o;
        EXPECT_EQ(r1.value(), 0.0);
        EXPECT_EQ(r1.has_value(), true);

        xtl::xoptional<double, bool> r2 = o - m;
        EXPECT_EQ(r2.value(), 0.0);
        EXPECT_EQ(r2.has_value(), true);

        xtl::xoptional<double, bool> r3 = m - m;
        EXPECT_EQ(r3.value(), 0.0);
        EXPECT_EQ(r3.has_value(), true);

        xtl::xoptional<double, bool> r4 = 4.2 - m;
        EXPECT_EQ(r4.value(), -1.0);
        EXPECT_EQ(r4.has_value(), true);

        xtl::xoptional<double, bool> r5 = m - 4.2;
        EXPECT_EQ(r5.value(), 1.0);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_mult)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        xtl::xoptional<double, bool> r1 = m * o;
        EXPECT_EQ(r1.value(), 25.);
        EXPECT_EQ(r1.has_value(), true);

        xtl::xoptional<double, bool> r2 = o * m;
        EXPECT_EQ(r2.value(), 25.);
        EXPECT_EQ(r2.has_value(), true);

        xtl::xoptional<double, bool> r3 = m * m;
        EXPECT_EQ(r3.value(), 25.);
        EXPECT_EQ(r3.has_value(), true);

        xtl::xoptional<double, bool> r4 = 4. * m;
        EXPECT_EQ(r4.value(), 20.);
        EXPECT_EQ(r4.has_value(), true);

        xtl::xoptional<double, bool> r5 = m * 4.;
        EXPECT_EQ(r5.value(), 20.);
        EXPECT_EQ(r5.has_value(), true);
    }

    TEST(xmasked_value, arithm_div)
    {
        double a = 5.0;
        bool b = true;

        auto o = optional_type(a, b);

        auto m = xmasked_value<double&, bool&>(a, b);

        xtl::xoptional<double, bool> r1 = m / o;
        EXPECT_EQ(r1.value(), 1.);
        EXPECT_EQ(r1.has_value(), true);

        xtl::xoptional<double, bool> r2 = o / m;
        EXPECT_EQ(r2.value(), 1.);
        EXPECT_EQ(r2.has_value(), true);

        xtl::xoptional<double, bool> r3 = m / m;
        EXPECT_EQ(r3.value(), 1.);
        EXPECT_EQ(r3.has_value(), true);

        xtl::xoptional<double, bool> r4 = 25. / m;
        EXPECT_EQ(r4.value(), 5.);
        EXPECT_EQ(r4.has_value(), true);

        xtl::xoptional<double, bool> r5 = m / 1.;
        EXPECT_EQ(r5.value(), 5.);
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

    TEST(xmasked_view, dimension)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.dimension(), masked_data.dimension());
    }

    TEST(xmasked_view, size)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.size(), masked_data.size());
    }

    TEST(xmasked_view, shape)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.shape(), masked_data.shape());
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    TEST(xmasked_view, access)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data(0, 0), 1.);
        EXPECT_EQ(masked_data(0, 1), 2.);
        EXPECT_EQ(masked_data(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data(2, 0), 7.);
        EXPECT_EQ(masked_data(2, 1), 8.);
        EXPECT_EQ(masked_data(2, 2), 9.);

#ifndef XTENSOR_ENABLE_ASSERT
        masked_data(3, 3);
#endif
    }

    TEST(xmasked_view, at)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data.at(0, 0), 1.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(2, 0), 7.);
        EXPECT_EQ(masked_data.at(2, 1), 8.);
        EXPECT_EQ(masked_data.at(2, 2), 9.);

        EXPECT_ANY_THROW(masked_data.at(3, 3));
    }

    TEST(xmasked_view, unchecked)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data.unchecked(0, 0), 1.);
        EXPECT_EQ(masked_data.unchecked(0, 1), 2.);
        EXPECT_EQ(masked_data.unchecked(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(2, 0), 7.);
        EXPECT_EQ(masked_data.unchecked(2, 1), 8.);
        EXPECT_EQ(masked_data.unchecked(2, 2), 9.);

#ifndef XTENSOR_ENABLE_ASSERT
        masked_data.unchecked(3, 3);
#endif
    }

    TEST(xmasked_view, access2)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto val = masked_data[{0, 0}];
        EXPECT_EQ(val, 1.);
        auto val2 = masked_data[{1, 0}];
        EXPECT_EQ(val2, xtl::missing<double>());

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data[index1], 1.);
        EXPECT_EQ(masked_data[index2], xtl::missing<double>());
    }

    TEST(xmasked_view, element)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data.element(index1.begin(), index1.end()), 1.);
        EXPECT_EQ(masked_data.element(index2.begin(), index2.end()), xtl::missing<double>());
    }

    TEST(xmasked_view, storage)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.storage().value(), masked_data.storage().value());
        EXPECT_NE(data.storage().has_value(), masked_data.storage().has_value());
    }

    TEST(xmasked_view, fill)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        masked_data.fill(2.);

        EXPECT_EQ(masked_data.at(0, 0), 2.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), 2.);
        EXPECT_EQ(masked_data.at(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(2, 0), 2.);
        EXPECT_EQ(masked_data.at(2, 1), 2.);
        EXPECT_EQ(masked_data.at(2, 2), 2.);
    }

    TEST(xmasked_view, non_optional_data)
    {
        xarray<double> data = {{ 1., 2., 3.},
                               { 4., 5., 6.},
                               { 7., 8., 9.}};
        xarray<bool> mask = {{ true,  true,  true},
                             { true, false, false},
                             { true, false,  true}};

        auto masked_data = masked_view(data, mask);

        EXPECT_EQ(masked_data(0, 0), 1.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), 3.);
        EXPECT_EQ(masked_data.unchecked(1, 0), 4.);
        EXPECT_EQ(masked_data.unchecked(1, 1), xtl::missing<double>());
        auto index1 = std::array<int, 2>({1, 2});
        EXPECT_EQ(masked_data[index1], xtl::missing<double>());

        masked_data = 3.65;
        xarray<double> expected = {{3.65, 3.65, 3.65},
                                   {3.65, 5.  , 6.  },
                                   {3.65, 8.  , 3.65}};
        EXPECT_EQ(masked_data.value(), expected);
    }
}
