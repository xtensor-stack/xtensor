/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "test_xsemantic.hpp"

namespace xt
{
    using vector_type = std::vector<int>;
    using adaptor_type = xarray_adaptor<std::vector<int>, layout::dynamic>;

    TEST(xadaptor_semantic, a_plus_b)
    {
        operation_tester<std::plus<>> tester;

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

    TEST(xadaptor_semantic, a_minus_b)
    {
        operation_tester<std::minus<>> tester;

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

    TEST(xadaptor_semantic, a_times_b)
    {
        operation_tester<std::multiplies<>> tester;

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

    TEST(xadaptor_semantic, a_divide_by_b)
    {
        operation_tester<std::divides<>> tester;

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

    TEST(xadaptor_semantic, a_plus_equal_b)
    {
        operation_tester<std::plus<>> tester;

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

    TEST(xadaptor_semantic, a_minus_equal_b)
    {
        operation_tester<std::minus<>> tester;

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

    TEST(xadaptor_semantic, a_times_equal_b)
    {
        operation_tester<std::multiplies<>> tester;

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

    TEST(xadaptor_semantic, a_divide_by_equal_b)
    {
        operation_tester<std::divides<>> tester;

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

