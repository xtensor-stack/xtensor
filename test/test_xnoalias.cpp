/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xview.hpp"

#include "test_common_macros.hpp"
#include "test_xsemantic.hpp"

namespace xt
{
    template <class C>
    class xnoalias : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

#define XNOALIAS_TEST_TYPES xarray_dynamic, xtensor_dynamic

    TEST_SUITE("xnoalias")
    {
        TEST_CASE_TEMPLATE("a_plus_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::plus<>, TypeParam> tester;

            SUBCASE("row_major + row_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a + tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major + column_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a + tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major + central_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a + tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major + unit_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a + tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_minus_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::minus<>, TypeParam> tester;

            SUBCASE("row_major - row_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a - tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major - column_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a - tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major - central_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a - tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major - unit_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a - tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_multiplies_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::multiplies<>, TypeParam> tester;

            SUBCASE("row_major * row_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a * tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major * column_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a * tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major * central_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a * tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major * unit_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a * tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_divides_by_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::divides<>, TypeParam> tester;

            SUBCASE("row_major / row_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a / tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major / column_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a / tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major / central_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a / tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major / unit_major")
            {
                TypeParam b(tester.ca.shape(), 0);
                noalias(b) = tester.a / tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_plus_equal_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::plus<>, TypeParam> tester;

            SUBCASE("row_major += row_major")
            {
                TypeParam b = tester.a;
                noalias(b) += tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major += column_major")
            {
                TypeParam b = tester.a;
                noalias(b) += tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major += central_major")
            {
                TypeParam b = tester.a;
                noalias(b) += tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major += unit_major")
            {
                TypeParam b = tester.a;
                noalias(b) += tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_minus_equal_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::minus<>, TypeParam> tester;

            SUBCASE("row_major -= row_major")
            {
                TypeParam b = tester.a;
                noalias(b) -= tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major -= column_major")
            {
                TypeParam b = tester.a;
                noalias(b) -= tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major -= central_major")
            {
                TypeParam b = tester.a;
                noalias(b) -= tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major -= unit_major")
            {
                TypeParam b = tester.a;
                noalias(b) -= tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_times_equal_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::multiplies<>, TypeParam> tester;

            SUBCASE("row_major *= row_major")
            {
                TypeParam b = tester.a;
                noalias(b) *= tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major *= column_major")
            {
                TypeParam b = tester.a;
                noalias(b) *= tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major *= central_major")
            {
                TypeParam b = tester.a;
                noalias(b) *= tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major *= unit_major")
            {
                TypeParam b = tester.a;
                noalias(b) *= tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE_TEMPLATE("a_divide_by_equal_b", TypeParam, XNOALIAS_TEST_TYPES)
        {
            operation_tester<std::divides<>, TypeParam> tester;

            SUBCASE("row_major /= row_major")
            {
                TypeParam b = tester.a;
                noalias(b) /= tester.ra;
                EXPECT_EQ(tester.res_rr, b);
            }

            SUBCASE("row_major /= column_major")
            {
                TypeParam b = tester.a;
                noalias(b) /= tester.ca;
                EXPECT_EQ(tester.res_rc, b);
            }

            SUBCASE("row_major /= central_major")
            {
                TypeParam b = tester.a;
                noalias(b) /= tester.cta;
                EXPECT_EQ(tester.res_rct, b);
            }

            SUBCASE("row_major /= unit_major")
            {
                TypeParam b = tester.a;
                noalias(b) /= tester.ua;
                EXPECT_EQ(tester.res_ru, b);
            }
        }

        TEST_CASE("scalar_ops")
        {
            xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            xarray<int> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            xarray<int> c = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

            xscalar<int> one(1), five(5), twelve(12), seven(7), bhalf(0b00001111), bxor(0b01001111);

            xt::noalias(a) += 1;
            b += 1;
            xt::noalias(c) += one;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) -= 5;
            b -= 5;
            xt::noalias(c) -= five;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) *= 12;
            b *= 12;
            xt::noalias(c) *= twelve;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) /= 7;
            b /= 7;
            xt::noalias(c) /= seven;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) %= 7;
            b %= 7;
            xt::noalias(c) %= seven;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) &= 0b00001111;
            b &= 0b00001111;
            xt::noalias(c) &= bhalf;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) |= 0b00001111;
            b |= 0b00001111;
            xt::noalias(c) |= bhalf;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) ^= 0b01001111;
            b ^= 0b01001111;
            xt::noalias(c) ^= bxor;
            EXPECT_EQ(a, b);
            EXPECT_EQ(a, c);

            xt::noalias(a) = 123;
            b = 123;

            EXPECT_EQ(a, b);
        }

        TEST_CASE("rvalue")
        {
            xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            xarray<int> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

            xt::noalias(xt::view(a, 1)) += 10;
            xt::view(b, 1) += 10;

            EXPECT_EQ(a, b);

            xt::noalias(xt::view(a, 1)) = 10;
            xt::view(b, 1) = 10;
            EXPECT_EQ(a, b);
        }
    }
}
