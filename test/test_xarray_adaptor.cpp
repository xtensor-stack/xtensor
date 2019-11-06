/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
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
    using adaptor_type = xarray_adaptor<vec_type, layout_type::dynamic>;

    TEST(xarray_adaptor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            vec_type v;
            adaptor_type a(v, rm.shape(), layout_type::row_major);
            compare_shape(a, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            vec_type v;
            adaptor_type a(v, cm.shape(), layout_type::column_major);
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
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
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
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.storage(), c.storage());
            adaptor_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xarray_adaptor, resize)
    {
        vec_type v;
        adaptor_type a(v);
        test_resize(a);
    }

    TEST(xarray_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        test_reshape(a);
    }

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xarray_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        test_access(a);
    }
#endif

    TEST(xarray_adaptor, unchecked)
    {
        vec_type v;
        adaptor_type a(v);
        test_unchecked(a);
    }

    TEST(xarray_adaptor, at)
    {
        vec_type v;
        adaptor_type a(v);
        test_at(a);
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
        using adaptor_rm = xarray_adaptor<vec_type, layout_type::row_major>;
        using adaptor_cm = xarray_adaptor<vec_type, layout_type::column_major>;
        adaptor_rm arm(v);
        adaptor_cm acm(v);
        test_iterator(arm, acm);
    }

    TEST(xarray_adaptor, fill)
    {
        vec_type v;
        adaptor_type a(v);
        test_fill(a);
    }

    TEST(xarray_adaptor, xiterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_xiterator(a);
    }

    TEST(xarray_adaptor, reverse_xiterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_reverse_xiterator(a);
    }

    TEST(xarray_adaptor, adapt_std_array)
    {
        std::array<double, 9> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        xt::xarray_adaptor<decltype(a)> ad(a, xt::dynamic_shape<std::size_t>{3, 3});
        EXPECT_EQ(ad(1, 1), 5.);
        ad = ad * 2;
        EXPECT_EQ(ad(1, 1), 10.);
    }

    TEST(xarray_adaptor, iterator_types)
    {
        using vec_type = std::vector<int>;
        using array_type = xarray_adaptor<vec_type>;
        using const_array_type = xarray_adaptor<const vec_type>;
        using iterator = vec_type::iterator;
        using const_iterator = vec_type::const_iterator;

        test_iterator_types<array_type, iterator, const_iterator>();
        test_iterator_types<const_array_type, const_iterator, const_iterator>();
    }
}
