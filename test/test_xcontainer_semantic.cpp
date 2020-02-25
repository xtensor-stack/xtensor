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
    template <class C>
    class container_semantic : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

    using testing_types = ::testing::Types<xarray_dynamic, xtensor_dynamic>;
    TYPED_TEST_SUITE(container_semantic, testing_types);

    TYPED_TEST(container_semantic, a_plus_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major + row_major");
            TypeParam b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            TypeParam b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            TypeParam b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            TypeParam b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_minus_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major - row_major");
            TypeParam b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            TypeParam b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            TypeParam b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            TypeParam b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_times_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major * row_major");
            TypeParam b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            TypeParam b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            TypeParam b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            TypeParam b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_divide_by_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major / row_major");
            TypeParam b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            TypeParam b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            TypeParam b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            TypeParam b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_and_b)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major & row_major");
            TypeParam b = tester.a & tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major & column_major");
            TypeParam b = tester.a & tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major & central_major");
            TypeParam b = tester.a & tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major & unit_major");
            TypeParam b = tester.a & tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_or_b)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major | row_major");
            TypeParam b = tester.a | tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major | column_major");
            TypeParam b = tester.a | tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major | central_major");
            TypeParam b = tester.a | tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major | unit_major");
            TypeParam b = tester.a | tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_xor_b)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major ^ row_major");
            TypeParam b = tester.a ^ tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major ^ column_major");
            TypeParam b = tester.a ^ tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major ^ central_major");
            TypeParam b = tester.a ^ tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major ^ unit_major");
            TypeParam b = tester.a ^ tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_plus_equal_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major += row_major");
            TypeParam b = tester.a;
            b += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            TypeParam b = tester.a;
            b += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            TypeParam b = tester.a;
            b += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            TypeParam b = tester.a;
            b += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_minus_equal_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major -= row_major");
            TypeParam b = tester.a;
            b -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            TypeParam b = tester.a;
            b -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            TypeParam b = tester.a;
            b -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            TypeParam b = tester.a;
            b -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_times_equal_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major *= row_major");
            TypeParam b = tester.a;
            b *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            TypeParam b = tester.a;
            b *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            TypeParam b = tester.a;
            b *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            TypeParam b = tester.a;
            b *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_divide_by_equal_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major /= row_major");
            TypeParam b = tester.a;
            b /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            TypeParam b = tester.a;
            b /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            TypeParam b = tester.a;
            b /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            TypeParam b = tester.a;
            b /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_and_equal_b)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major &= row_major");
            TypeParam b = tester.a;
            b &= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major &= column_major");
            TypeParam b = tester.a;
            b &= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major &= central_major");
            TypeParam b = tester.a;
            b &= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major &= unit_major");
            TypeParam b = tester.a;
            b &= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_or_equal_b)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major |= row_major");
            TypeParam b = tester.a;
            b |= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major |= column_major");
            TypeParam b = tester.a;
            b |= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major |= central_major");
            TypeParam b = tester.a;
            b |= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major |= unit_major");
            TypeParam b = tester.a;
            b |= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, a_bitwise_xor_equal_b)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major ^= row_major");
            TypeParam b = tester.a;
            b ^= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major ^= column_major");
            TypeParam b = tester.a;
            b ^= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major ^= central_major");
            TypeParam b = tester.a;
            b ^= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major ^= unit_major");
            TypeParam b = tester.a;
            b ^= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_plus_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major + row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_minus_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major - row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_multiplies_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major * row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_divides_by_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major / row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_bitwise_and_b)
    {
        operation_tester<std::bit_and<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major & row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major & column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major & central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major & unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a & tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_bitwise_or_b)
    {
        operation_tester<std::bit_or<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major | row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major | column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major | central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major | unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a | tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(container_semantic, assign_a_bitwise_xor_b)
    {
        operation_tester<std::bit_xor<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major ^ row_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major ^ column_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major ^ central_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major ^ unit_major");
            TypeParam b(tester.ca.shape(), 0);
            b = tester.a ^ tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }
}
