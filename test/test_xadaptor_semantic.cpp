/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "test_xsemantic.hpp"

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
    class adaptor_semantic : public ::testing::Test
    {
    public:
        using container_type = C;
        using adaptor_type = get_test_adaptor_t<C>;
    };

    using testing_types = ::testing::Types<xarray_dynamic, xtensor_dynamic>;
    TYPED_TEST_CASE(adaptor_semantic, testing_types);

    TYPED_TEST(adaptor_semantic, a_plus_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;
        {
            SCOPED_TRACE("row_major + row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_minus_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major - row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_times_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major * row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_divide_by_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major / row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_plus_equal_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major += row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_minus_equal_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major -= row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_times_equal_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major *= row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(adaptor_semantic, a_divide_by_equal_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;
        using adaptor_type = typename TestFixture::adaptor_type;

        {
            SCOPED_TRACE("row_major /= row_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            vector_type v;
            adaptor_type b(v);
            b = tester.a;
            b /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }
}
