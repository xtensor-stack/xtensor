/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
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
    using vec_type = std::vector<int>;
    using adaptor_type = xtensor_adaptor<vec_type, 3, layout_type::dynamic>;
    using storage_type = std::array<std::size_t, 3>;

    TEST(xtensor_adaptor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<storage_type> rm;
            vec_type v;
            adaptor_type a(v, rm.shape(), layout_type::row_major);
            compare_shape(a, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<storage_type> cm;
            vec_type v;
            adaptor_type a(v, cm.shape(), layout_type::column_major);
            compare_shape(a, cm);
        }
    }

    TEST(xtensor_adaptor, strided_constructor)
    {
        central_major_result<storage_type> cmr;
        vec_type v;
        adaptor_type a(v, cmr.shape(), cmr.strides());
        compare_shape(a, cmr);
    }

    TEST(xtensor_adaptor, copy_semantic)
    {
        central_major_result<storage_type> res;
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
            row_major_result<storage_type> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xtensor_adaptor, move_semantic)
    {
        central_major_result<storage_type> res;
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
            row_major_result<storage_type> r;
            vec_type v2(r.size(), 0);
            adaptor_type c(v2, r.shape());
            EXPECT_NE(a.storage(), c.storage());
            adaptor_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xtensor_adaptor, resize)
    {
        vec_type v;
        adaptor_type a(v);
        test_resize<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        test_reshape<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        test_access<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, unchecked)
    {
        vec_type v;
        adaptor_type a(v);
        test_unchecked<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, at)
    {
        vec_type v;
        adaptor_type a(v);
        test_at<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, indexed_access)
    {
        vec_type v;
        adaptor_type a(v);
        test_indexed_access<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, broadcast_shape)
    {
        vec_type v;
        xtensor_adaptor<vec_type, 4> a(v);
        test_broadcast(a);
    }

    TEST(xtensor_adaptor, iterator)
    {
        vec_type v;
        using adaptor_rm = xtensor_adaptor<vec_type, 3, layout_type::row_major>;
        using adaptor_cm = xtensor_adaptor<vec_type, 3, layout_type::column_major>;
        adaptor_rm arm(v);
        adaptor_cm acm(v);
        test_iterator<adaptor_rm, adaptor_cm, storage_type>(arm, acm);
    }

    TEST(xtensor_adaptor, fill)
    {
        vec_type v;
        xtensor_adaptor<vec_type, 2> a(v);
        test_fill(a);
    }

    TEST(xtensor_adaptor, xiterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_xiterator<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, reverse_xiterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_reverse_xiterator<adaptor_type, storage_type>(a);
    }

    TEST(xtensor_adaptor, adapt_std_array)
    {
        std::array<double, 9> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        xtensor_adaptor<decltype(a), 2> ad(a, {3, 3});
        EXPECT_EQ(ad(1, 1), 5.);
        ad = ad * 2;
        EXPECT_EQ(ad(1, 1), 10.);
    }

    TEST(xtensor_adaptor, iterator_types)
    {
        using vec_type = std::vector<int>;
        using tensor_type = xtensor_adaptor<vec_type, 2>;
        using const_tensor_type = xtensor_adaptor<const vec_type, 2>;
        using iterator = vec_type::iterator;
        using const_iterator = vec_type::const_iterator;

        test_iterator_types<tensor_type, iterator, const_iterator>();
        test_iterator_types<const_tensor_type, const_iterator, const_iterator>();
    }
}
