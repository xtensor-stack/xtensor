/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "test_common.hpp"

namespace xt
{
    using container_type = std::array<std::size_t, 3>;
    using xtensor_type = xtensor<int, 3, layout::dynamic>;
    TEST(xtensor, initializer_constructor)
    {
        xtensor_type t 
          {{{0, 1, 2}, 
            {3, 4, 5}, 
            {6, 7, 8}}, 
           {{9, 10, 11}, 
            {12, 13, 14}, 
            {15, 16, 17}}}; 
        EXPECT_EQ(t.dimension(), 3);
        EXPECT_EQ(t(0, 0, 1), 1);
        EXPECT_EQ(t.shape()[0], 2);
    }

    TEST(xtensor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<container_type> rm;
            xtensor_type ra(rm.m_shape);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<container_type> cm;
            xtensor_type ca(cm.m_shape, layout::column_major);
            compare_shape(ca, cm);
        }
    }

    TEST(xtensor, strided_constructor)
    {
        central_major_result<container_type> cmr;
        xtensor_type cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xtensor, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<container_type> rm;
            int value = 2;
            xtensor_type ra(rm.m_shape, value);
            compare_shape(ra, rm);
            xtensor_type::container_type vec(ra.size(), value);
            EXPECT_EQ(ra.data(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<container_type> cm;
            int value = 2;
            xtensor_type ca(cm.m_shape, value, layout::column_major);
            compare_shape(ca, cm);
            xtensor_type::container_type vec(ca.size(), value);
            EXPECT_EQ(ca.data(), vec);
        }
    }

    TEST(xtensor, strided_valued_constructor)
    {
        central_major_result<container_type> cmr;
        int value = 2;
        xtensor_type cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        xtensor_type::container_type vec(cma.size(), value);
        EXPECT_EQ(cma.data(), vec);
    }

    TEST(xtensor, copy_semantic)
    {
        central_major_result<container_type> res;
        int value = 2;
        xtensor_type a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            xtensor_type b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<container_type> r;
            xtensor_type c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xtensor, move_semantic)
    {
        central_major_result<container_type> res;
        int value = 2;
        xtensor_type a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            xtensor_type tmp(a);
            xtensor_type b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<container_type> r;
            xtensor_type c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            xtensor_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xtensor, reshape)
    {
        xtensor_type a;
        test_reshape<xtensor_type, container_type>(a);
    }

    TEST(xtensor, transpose)
    {
        xtensor_type a;
        test_transpose<xtensor_type, container_type>(a);
    }

    TEST(xtensor, access)
    {
        xtensor_type a;
        test_access<xtensor_type, container_type>(a);
    }

    TEST(xtensor, indexed_access)
    {
        xtensor_type a;
        test_indexed_access<xtensor_type, container_type>(a);
    }

    TEST(xtensor, broadcast_shape)
    {
        xtensor<int, 4> a;
        test_broadcast(a);
    }

    TEST(xtensor, iterator)
    {
        xtensor_type a;
        test_iterator<xtensor_type, container_type>(a);
    }

    TEST(xtensor, zerod)
    {
        xtensor_type a;
        EXPECT_EQ(0, a());
    }
}
