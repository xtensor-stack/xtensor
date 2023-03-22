/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/xnoalias.hpp"

#include "test_common_macros.hpp"
#include "test_xsemantic.hpp"

namespace xt
{

    template <class T1, class T2>
    inline bool full_equal(const T1& a1, const T2& a2)
    {
        return (a1.strides() == a2.strides()) && (a1 == a2);
    }

    template <class C>
    class scalar_semantic : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

#define SCALAR_SEMANTIC_TEST_TYPES xarray_dynamic, xtensor_dynamic

    TEST_SUITE("scalar_semantic")
    {
        TEST_CASE_TEMPLATE("a_plus_equal_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::plus<>, TypeParam> tester;

            SUBCASE("row_major += scalar")
            {
                TypeParam a = tester.ra;
                a += tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major += scalar")
            {
                TypeParam a = tester.ca;
                a += tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major += scalar")
            {
                TypeParam a = tester.cta;
                a += tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major += scalar")
            {
                TypeParam a = tester.ua;
                a += tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("a_minus_equal_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::minus<>, TypeParam> tester;

            SUBCASE("row_major -= scalar")
            {
                TypeParam a = tester.ra;
                a -= tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major -= scalar")
            {
                TypeParam a = tester.ca;
                a -= tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major -= scalar")
            {
                TypeParam a = tester.cta;
                a -= tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major -= scalar")
            {
                TypeParam a = tester.ua;
                a -= tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("a_times_equal_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::multiplies<>, TypeParam> tester;

            SUBCASE("row_major *= scalar")
            {
                TypeParam a = tester.ra;
                a *= tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major *= scalar")
            {
                TypeParam a = tester.ca;
                a *= tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major *= scalar")
            {
                TypeParam a = tester.cta;
                a *= tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major *= scalar")
            {
                TypeParam a = tester.ua;
                a *= tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("a_divide_by_equal_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::divides<>, TypeParam> tester;

            SUBCASE("row_major /= scalar")
            {
                TypeParam a = tester.ra;
                a /= tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major /= scalar")
            {
                TypeParam a = tester.ca;
                a /= tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major /= scalar")
            {
                TypeParam a = tester.cta;
                a /= tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major /= scalar")
            {
                TypeParam a = tester.ua;
                a /= tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("assign_a_plus_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::plus<>, TypeParam> tester;

            SUBCASE("row_major + scalar")
            {
                TypeParam a(tester.ra.shape(), tester.ra.strides(), 0);
                noalias(a) = tester.ra + tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major + scalar")
            {
                TypeParam a(tester.ca.shape(), tester.ca.strides(), 0);
                noalias(a) = tester.ca + tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major + scalar")
            {
                TypeParam a(tester.cta.shape(), tester.cta.strides(), 0);
                noalias(a) = tester.cta + tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major + scalar")
            {
                TypeParam a(tester.ua.shape(), tester.ua.strides(), 0);
                noalias(a) = tester.ua + tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("assign_a_minus_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::minus<>, TypeParam> tester;

            SUBCASE("row_major - scalar")
            {
                TypeParam a(tester.ra.shape(), tester.ra.strides(), 0);
                noalias(a) = tester.ra - tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major - scalar")
            {
                TypeParam a(tester.ca.shape(), tester.ca.strides(), 0);
                noalias(a) = tester.ca - tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major - scalar")
            {
                TypeParam a(tester.cta.shape(), tester.cta.strides(), 0);
                noalias(a) = tester.cta - tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major - scalar")
            {
                TypeParam a(tester.ua.shape(), tester.ua.strides(), 0);
                noalias(a) = tester.ua - tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("assign_a_times_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::multiplies<>, TypeParam> tester;

            SUBCASE("row_major * scalar")
            {
                TypeParam a(tester.ra.shape(), tester.ra.strides(), 0);
                noalias(a) = tester.ra * tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major * scalar")
            {
                TypeParam a(tester.ca.shape(), tester.ca.strides(), 0);
                noalias(a) = tester.ca * tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major * scalar")
            {
                TypeParam a(tester.cta.shape(), tester.cta.strides(), 0);
                noalias(a) = tester.cta * tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major * scalar")
            {
                TypeParam a(tester.ua.shape(), tester.ua.strides(), 0);
                noalias(a) = tester.ua * tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }

        TEST_CASE_TEMPLATE("assign_a_divide_by_b", TypeParam, SCALAR_SEMANTIC_TEST_TYPES)
        {
            scalar_operation_tester<std::divides<>, TypeParam> tester;

            SUBCASE("row_major / scalar")
            {
                TypeParam a(tester.ra.shape(), tester.ra.strides(), 0);
                noalias(a) = tester.ra / tester.b;
                EXPECT_TRUE(full_equal(tester.res_r, a));
            }

            SUBCASE("column_major / scalar")
            {
                TypeParam a(tester.ca.shape(), tester.ca.strides(), 0);
                noalias(a) = tester.ca / tester.b;
                EXPECT_TRUE(full_equal(tester.res_c, a));
            }

            SUBCASE("central_major / scalar")
            {
                TypeParam a(tester.cta.shape(), tester.cta.strides(), 0);
                noalias(a) = tester.cta / tester.b;
                EXPECT_TRUE(full_equal(tester.res_ct, a));
            }

            SUBCASE("unit_major / scalar")
            {
                TypeParam a(tester.ua.shape(), tester.ua.strides(), 0);
                noalias(a) = tester.ua / tester.b;
                EXPECT_TRUE(full_equal(tester.res_u, a));
            }
        }
    }

#undef SCALAR_SEMANTIC_TEST_TYPES
}
