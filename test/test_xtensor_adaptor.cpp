/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/xtensor.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    using vec_type = std::vector<int>;
    using adaptor_type = xtensor_adaptor<vec_type, 3, layout_type::dynamic>;
    using storage_type = std::array<std::size_t, 3>;

    TEST(xtensor_adaptor, shaped_constructor)
    {
        SUBCASE("row_major constructor")
        {
            row_major_result<storage_type> rm;
            vec_type v;
            adaptor_type a(v, rm.shape(), layout_type::row_major);
            compare_shape(a, rm);
        }

        SUBCASE("column_major constructor")
        {
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

        SUBCASE("copy constructor")
        {
            adaptor_type b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        SUBCASE("assignment operator")
        {
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

        SUBCASE("move constructor")
        {
            adaptor_type tmp(a);
            adaptor_type b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        SUBCASE("move assignment")
        {
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

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xtensor_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        test_access<adaptor_type, storage_type>(a);
    }
#endif

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

    namespace xt_shared
    {
        template <class T, std::size_t N, layout_type L = layout_type::dynamic>
        using xtensor_buffer = xtensor_adaptor<xbuffer_adaptor<T, xt::smart_ownership, std::shared_ptr<T[]>>, N, L>;
    }

    TEST(xtensor_shared_buffer, share_shared_pointer)
    {
        using T = double;
        using xtensor_type = xt_shared::xtensor_buffer<double, 1>;
        using storage_type = xtensor_type::storage_type;
        using inner_shape_type = typename xtensor_type::inner_shape_type;
        using inner_strides_type = typename xtensor_type::inner_strides_type;
        auto shape = xtensor_type::shape_type{100};
        auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        auto data = std::shared_ptr<T[]>(new T[size]);
        auto data2 = std::shared_ptr<T[]>(new T[size]);
        inner_shape_type inner_shape = shape;
        inner_shape_type inner_shape2 = shape;
        inner_strides_type inner_strides;
        xt::compute_strides(inner_shape, XTENSOR_DEFAULT_LAYOUT, inner_strides);

        // Create storage with shared ownership
        storage_type s(data.get(), size, data);
        storage_type s2(data.get(), size, data);
        // s3 has no shared ownership
        storage_type s3(data2.get(), size, data);
        // Now create the respective tensors
        xtensor_type x(std::move(s), inner_shape_type(shape), inner_strides_type(inner_strides));
        xtensor_type x2(std::move(s2), inner_shape_type(shape), inner_strides_type(inner_strides));
        xtensor_type x3(std::move(s3), inner_shape_type(shape), inner_strides_type(inner_strides));

        // Initialize both tensors (x & x3) to zero
        x = xt::broadcast(double(0), {size});
        x3 = xt::broadcast(double(0), {size});

        // Assign another shared memory tensor to x shared memory
        xtensor_type y = x;

        // Modify the value in x
        x(0) = 1.0;

        // Use x2 now
        y = x2;

        // We do get 1.0 in y, because x and y share the same memory
        EXPECT_EQ(y(0), 1.0);

        // Now assign x3 memory to y (i.e. unshare it with x)
        y = x3;
        // Change value in x2
        x2(0) = 2.0;
        // We do not get 2.0 in y, because x2 and y do not share the same memory
        EXPECT_EQ(y(0), 0.0);
        // We do get 2.0 in x2
        EXPECT_EQ(x2(0), 2.0);
    }
}
