/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "test_common.hpp"
#include "test_xsemantic.hpp"

TEST_SUITE_BEGIN("adaptor_semantic");

namespace xt
{
    using vector_type = std::vector<int>;
    using array_adaptor_type = xarray_adaptor<std::vector<int>, layout_type::dynamic>;
    using tensor_adaptor_type = xtensor_adaptor<std::vector<int>, 3, layout_type::dynamic>;

    template <class C>
    struct get_test_adaptor;

    template <>
    struct get_test_adaptor<xarray_dynamic>
    {
        using type = array_adaptor_type;
    };

    template <>
    struct get_test_adaptor<xtensor_dynamic>
    {
        using type = tensor_adaptor_type;
    };

    template <class C>
    using get_test_adaptor_t = typename get_test_adaptor<C>::type;

    template <class C>
    class adaptor_semantic
    {
    public:

        using container_type = C;
        using adaptor_type = get_test_adaptor_t<C>;
    };

#define ADAPTOR_SEMANTIC_TYPES xarray_dynamic, xtensor_dynamic

    TEST_CASE_TEMPLATE("xsimd_info", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
#if defined(XTENSOR_USE_XSIMD)
        std::cout << "Built with XSIMD" << std::endl;
        std::cout << " arch " << xsimd::default_arch::name() << std::endl;
#else
        std::cout << "Built without XSIMD" << std::endl;
#endif
    }

    TEST_CASE_TEMPLATE("a_plus_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::plus<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;
        SUBCASE("row_major + row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major + column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major + central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major + unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_minus_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::minus<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major - row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major - column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major - central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major - unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_times_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major * row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major * column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major * central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major * unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_divide_by_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::divides<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major / row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major / column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major / central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major / unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_plus_equal_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::plus<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major += row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major += column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major += central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major += unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_minus_equal_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::minus<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major -= row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major -= column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major -= central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major -= unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_times_equal_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major *= row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major *= column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major *= central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major *= unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST_CASE_TEMPLATE("a_divide_by_equal_b", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        operation_tester<std::divides<>, TypeParam> tester;
        using adaptor_type = typename adaptor_semantic<TypeParam>::adaptor_type;

        SUBCASE("row_major /= row_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        SUBCASE("row_major /= column_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        SUBCASE("row_major /= central_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        SUBCASE("row_major /= unit_major")
        {
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

#undef ADAPTOR_SEMANTIC_TYPES
}

TEST_SUITE_END();
