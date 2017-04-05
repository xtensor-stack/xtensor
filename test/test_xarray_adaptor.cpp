/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "test_common.hpp"

namespace xt
{
    using vec_type = std::vector<int>;
    using adaptor_type = xarray_adaptor<vec_type>;

    TEST(xarray_adaptor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            vec_type v;
            adaptor_type a(v, rm.shape());
            compare_shape(a, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            vec_type v;
            xarray_adaptor<vec_type, layout::column_major> a(v, cm.shape());
            compare_shape(a, cm);
        }
    }
    
    TEST(xarray_adaptor, strided_constructor)
    {
        central_major_result<> cmr;
        vec_type v;
        adaptor_type a(v, cmr.shape(), cmr.strides());
        compare_shape(a, cmr);
    }

    TEST(xarray_adaptor, copy_semantic)
    {
        central_major_result<> res;
        int value = 2;
        vec_type v(res.size(), value);
        adaptor_type a(v, res.shape(), res.strides());

        {
            SCOPED_TRACE("copy constructor");
            adaptor_type b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xarray_adaptor, move_semantic)
    {
        central_major_result<> res;
        int value = 2;
        vec_type v(res.size(), value);
        adaptor_type a(v, res.shape(), res.strides());

        {
            SCOPED_TRACE("move constructor");
            adaptor_type tmp(a);
            adaptor_type b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.data(), c.data());
            adaptor_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xarray_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        test_reshape(a);
    }

    TEST(xarray_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        test_access(a);
    }

    TEST(xarray_adaptor, indexed_access)
    {
        vec_type v;
        adaptor_type a(v);
        test_indexed_access(a);
    }

    TEST(xarray_adaptor, broadcast_shape)
    {
        vec_type v;
        adaptor_type a(v);
        test_broadcast(a);
        test_broadcast2(a);
    }

    TEST(xarray_adaptor, iterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_iterator(a);
    }

    TEST(xarray_adaptor, move_reshape)
    {
        vec_type adapted = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::vector<size_t> adapted_shape = { 2, 6 };
        xarray_adaptor<vec_type> a(adapted, adapted_shape);
        std::vector<size_t> shape = { 2, 2, 3 };
        xarray_adaptor<vec_type> b = move_reshape(a, shape);
        EXPECT_EQ(1, b(0, 0, 0));
        EXPECT_EQ(5, b(0, 1, 1));
        EXPECT_EQ(12, b(1, 1, 2));
    }
}

