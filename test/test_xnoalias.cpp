/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "test_xsemantic.hpp"

namespace xt
{
    TEST(xnoalias, a_plus_b)
    {
        operation_tester<std::plus<>> tester;

        {
            SCOPED_TRACE("row_major + row_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_minus_b)
    {
        operation_tester<std::minus<>> tester;

        {
            SCOPED_TRACE("row_major - row_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_multiplies_b)
    {
        operation_tester<std::multiplies<>> tester;

        {
            SCOPED_TRACE("row_major * row_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_divides_by_b)
    {
        operation_tester<std::divides<>> tester;

        {
            SCOPED_TRACE("row_major / row_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            xarray<int> b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_plus_equal_b)
    {
        operation_tester<std::plus<>> tester;

        {
            SCOPED_TRACE("row_major += row_major");
            xarray<int> b = tester.a;
            noalias(b) += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            xarray<int> b = tester.a;
            noalias(b) += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            xarray<int> b = tester.a;
            noalias(b) += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            xarray<int> b = tester.a;
            noalias(b) += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_minus_equal_b)
    {
        operation_tester<std::minus<>> tester;

        {
            SCOPED_TRACE("row_major -= row_major");
            xarray<int> b = tester.a;
            noalias(b) -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            xarray<int> b = tester.a;
            noalias(b) -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            xarray<int> b = tester.a;
            noalias(b) -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            xarray<int> b = tester.a;
            noalias(b) -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_times_equal_b)
    {
        operation_tester<std::multiplies<>> tester;

        {
            SCOPED_TRACE("row_major *= row_major");
            xarray<int> b = tester.a;
            noalias(b) *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            xarray<int> b = tester.a;
            noalias(b) *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            xarray<int> b = tester.a;
            noalias(b) *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            xarray<int> b = tester.a;
            noalias(b) *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, a_divide_by_equal_b)
    {
        operation_tester<std::divides<>> tester;

        {
            SCOPED_TRACE("row_major /= row_major");
            xarray<int> b = tester.a;
            noalias(b) /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            xarray<int> b = tester.a;
            noalias(b) /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            xarray<int> b = tester.a;
            noalias(b) /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            xarray<int> b = tester.a;
            noalias(b) /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }
}

