/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/containers/xadapt.hpp"
#include "xtensor/core/xnoalias.hpp"

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

    TEST_CASE_TEMPLATE("scalar_assignment_with_strides", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        using shape_type = typename TypeParam::shape_type;
        using strides_type = typename TypeParam::strides_type;
        shape_type shape = {2, 1, 3};
        strides_type strides;
        std::vector<int> data;
        std::vector<int> expected;
        std::vector<int> doubled;

        SUBCASE("padded rows")
        {
            strides = {4, 0, 1};
            data = {2, 4, 6, -99, 8, 10, 12, -99};
            expected = {4, 6, 8, -99, 10, 12, 14, -99};
            doubled = {4, 8, 12, -99, 16, 20, 24, -99};
        }
        SUBCASE("padded columns")
        {
            strides = {1, 0, 3};
            data = {2, 4, -99, 6, 8, -99, 10, 12, -99};
            expected = {4, 6, -99, 8, 10, -99, 12, 14, -99};
            doubled = {4, 8, -99, 12, 16, -99, 20, 24, -99};
        }

        auto original = data;
        auto a = adapt(data.data(), 6, no_ownership(), shape, strides);
        a += 2;
        EXPECT_EQ(data, expected);
        noalias(a) -= 2;
        EXPECT_EQ(data, original);
        a *= 2;
        EXPECT_EQ(data, doubled);
        noalias(a) /= 2;
        EXPECT_EQ(data, original);
        EXPECT_EQ(a.shape(), shape);
        EXPECT_EQ(a.strides(), strides);
    }

    TEST_CASE_TEMPLATE("scalar_assignment_empty_strided", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        using shape_type = typename TypeParam::shape_type;
        using strides_type = typename TypeParam::strides_type;
        std::vector<int> data = {-99};
        shape_type shape = {2, 0, 3};
        strides_type strides = {4, 0, 1};
        auto a = adapt(data.data(), 0, no_ownership(), shape, strides);
        a += 2;
        EXPECT_EQ(data, std::vector<int>{-99});
    }

    TEST_CASE_TEMPLATE("scalar_assignment_aliased_scalar", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        using shape_type = typename TypeParam::shape_type;
        using strides_type = typename TypeParam::strides_type;
        shape_type shape = {2, 1, 2};
        strides_type strides;
        std::vector<int> data;
        std::vector<int> expected;
        std::size_t scalar_index = 0;
        std::size_t size = 4;

        SUBCASE("contiguous")
        {
            strides = {2, 0, 1};
            data = {1, 2, 3, 4};
            expected = {2, 4, 5, 6};
        }
        SUBCASE("contiguous columns")
        {
            strides = {1, 0, 2};
            data = {1, 2, 3, 4};
            expected = {3, 4, 7, 8};
            scalar_index = 1;
        }
        SUBCASE("contiguous axis permutation")
        {
            shape = {2, 3, 2};
            strides = {1, 4, 2};
            data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            expected = {8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26};
            scalar_index = 6;
            size = 12;
        }
        SUBCASE("padded")
        {
            strides = {3, 0, 1};
            data = {1, 2, -99, 3, 4, -99};
            expected = {2, 4, -99, 5, 6, -99};
        }

        auto a = adapt(data.data(), size, no_ownership(), shape, strides);
        a += data[scalar_index];
        EXPECT_EQ(data, expected);
    }

    TEST_CASE_TEMPLATE("scalar_assignment_noncopyable_functor", TypeParam, ADAPTOR_SEMANTIC_TYPES)
    {
        struct counted_add
        {
            counted_add() = default;
            counted_add(const counted_add&) = delete;

            int operator()(int lhs, int rhs)
            {
                return lhs + rhs + ++calls;
            }

            int calls = 0;
        };

        using shape_type = typename TypeParam::shape_type;
        using strides_type = typename TypeParam::strides_type;
        shape_type shape = {2, 1, 2};
        strides_type strides = {3, 0, 1};
        std::vector<int> data = {1, 2, -99, 3, 4, -99};
        std::vector<int> expected;
        std::size_t size = 0;

        SUBCASE("padded")
        {
            size = 4;
            expected = {4, 6, -99, 8, 10, -99};
        }
        SUBCASE("contiguous columns")
        {
            strides = {1, 0, 2};
            size = 4;
            data = {1, 2, 3, 4};
            expected = {4, 6, 8, 10};
        }
        SUBCASE("contiguous axis permutation")
        {
            shape = {2, 3, 2};
            strides = {1, 4, 2};
            size = 12;
            data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            expected = {4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26};
        }
        SUBCASE("empty")
        {
            shape = {2, 0, 2};
            size = 0;
            expected = data;
        }

        auto a = adapt(data.data(), size, no_ownership(), shape, strides);
        counted_add f;
        scalar_computed_assign(a, 2, f);
        EXPECT_EQ(data, expected);
        EXPECT_EQ(f.calls, static_cast<int>(size));
    }

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
