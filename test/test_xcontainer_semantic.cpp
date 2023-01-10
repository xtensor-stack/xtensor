/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "test_common_macros.hpp"
#include "test_xsemantic.hpp"

TEST_SUITE_BEGIN("container_semantic");

namespace xt
{
    template <class C>
    class container_semantic : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

#define CONTAINER_SEMANTIC_TYPES xarray_dynamic, xtensor_dynamic

    TEST_CASE_TEMPLATE("a_plus_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        SUBCASE("row_major + row_major")
        {
            TypeParam b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major + column_major")
        {
            TypeParam b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major + central_major")
        {
            TypeParam b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major + unit_major")
        {
            TypeParam b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_minus_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        SUBCASE("row_major - row_major")
        {
            TypeParam b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major - column_major")
        {
            TypeParam b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major - central_major")
        {
            TypeParam b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major - unit_major")
        {
            TypeParam b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_times_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        SUBCASE("row_major * row_major")
        {
            TypeParam b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major * column_major")
        {
            TypeParam b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major * central_major")
        {
            TypeParam b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major * unit_major")
        {
            TypeParam b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_divide_by_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        SUBCASE("row_major / row_major")
        {
            TypeParam b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major / column_major")
        {
            TypeParam b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major / central_major")
        {
            TypeParam b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major / unit_major")
        {
            TypeParam b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_and_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        SUBCASE("row_major & row_major")
        {
            TypeParam b = tester.a & tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major & column_major")
        {
            TypeParam b = tester.a & tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major & central_major")
        {
            TypeParam b = tester.a & tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major & unit_major")
        {
            TypeParam b = tester.a & tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_or_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        SUBCASE("row_major | row_major")
        {
            TypeParam b = tester.a | tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major | column_major")
        {
            TypeParam b = tester.a | tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major | central_major")
        {
            TypeParam b = tester.a | tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major | unit_major")
        {
            TypeParam b = tester.a | tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_xor_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        SUBCASE("row_major ^ row_major")
        {
            TypeParam b = tester.a ^ tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major ^ column_major")
        {
            TypeParam b = tester.a ^ tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major ^ central_major")
        {
            TypeParam b = tester.a ^ tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major ^ unit_major")
        {
            TypeParam b = tester.a ^ tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_plus_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        SUBCASE("row_major += row_major")
        {
            TypeParam b = tester.a;
            b += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major += column_major")
        {
            TypeParam b = tester.a;
            b += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major += central_major")
        {
            TypeParam b = tester.a;
            b += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major += unit_major")
        {
            TypeParam b = tester.a;
            b += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_minus_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        SUBCASE("row_major -= row_major")
        {
            TypeParam b = tester.a;
            b -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major -= column_major")
        {
            TypeParam b = tester.a;
            b -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major -= central_major")
        {
            TypeParam b = tester.a;
            b -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major -= unit_major")
        {
            TypeParam b = tester.a;
            b -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_times_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        SUBCASE("row_major *= row_major")
        {
            TypeParam b = tester.a;
            b *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major *= column_major")
        {
            TypeParam b = tester.a;
            b *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major *= central_major")
        {
            TypeParam b = tester.a;
            b *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major *= unit_major")
        {
            TypeParam b = tester.a;
            b *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_divide_by_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        SUBCASE("row_major /= row_major")
        {
            TypeParam b = tester.a;
            b /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major /= column_major")
        {
            TypeParam b = tester.a;
            b /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major /= central_major")
        {
            TypeParam b = tester.a;
            b /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major /= unit_major")
        {
            TypeParam b = tester.a;
            b /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_and_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        SUBCASE("row_major &= row_major")
        {
            TypeParam b = tester.a;
            b &= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major &= column_major")
        {
            TypeParam b = tester.a;
            b &= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major &= central_major")
        {
            TypeParam b = tester.a;
            b &= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major &= unit_major")
        {
            TypeParam b = tester.a;
            b &= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_or_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        SUBCASE("row_major |= row_major")
        {
            TypeParam b = tester.a;
            b |= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major |= column_major")
        {
            TypeParam b = tester.a;
            b |= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major |= central_major")
        {
            TypeParam b = tester.a;
            b |= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major |= unit_major")
        {
            TypeParam b = tester.a;
            b |= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_bitwise_xor_equal_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        SUBCASE("row_major ^= row_major")
        {
            TypeParam b = tester.a;
            b ^= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major ^= column_major")
        {
            TypeParam b = tester.a;
            b ^= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major ^= central_major")
        {
            TypeParam b = tester.a;
            b ^= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major ^= unit_major")
        {
            TypeParam b = tester.a;
            b ^= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_plus_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        SUBCASE("row_major + row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major + column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major + central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major + unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_minus_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        SUBCASE("row_major - row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major - column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major - central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major - unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_multiplies_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        SUBCASE("row_major * row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major * column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major * central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major * unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_divides_by_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        SUBCASE("row_major / row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major / column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major / central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major / unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_bitwise_and_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        SUBCASE("row_major & row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major & column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major & central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major & unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_bitwise_or_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        SUBCASE("row_major | row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major | column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major | central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major | unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("assign_a_bitwise_xor_b", TypeParam, xarray_dynamic, xtensor_dynamic)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        SUBCASE("row_major ^ row_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major ^ column_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major ^ central_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major ^ unit_major")
        {
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

#undef CONTAINER_SEMANTIC_TYPES
}

TEST_SUITE_END();
