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
    using container_type = std::array<std::size_t, 3>;
    using xtensor_dynamic = xtensor<int, 3, layout_type::dynamic>;

    TEST(xtensor, initializer_constructor)
    {
        xtensor_dynamic t
          {{{0, 1, 2},
            {3, 4, 5},
            {6, 7, 8}},
           {{9, 10, 11},
            {12, 13, 14},
            {15, 16, 17}}};
        EXPECT_EQ(t.dimension(), size_t(3));
        EXPECT_EQ(t(0, 0, 1), 1);
        EXPECT_EQ(t.shape()[0], size_t(2));
    }

    TEST(xtensor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<container_type> rm;
            xtensor_dynamic ra(rm.m_shape, layout_type::row_major);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<container_type> cm;
            xtensor_dynamic ca(cm.m_shape, layout_type::column_major);
            compare_shape(ca, cm);
        }

        {
            SCOPED_TRACE("from shape");
            std::array<std::size_t, 3> shp = {5, 4, 2};
            std::vector<std::size_t> shp_as_vec = {5, 4, 2};
            auto ca = xtensor<int, 3>::from_shape({3, 2, 1});
            auto cb = xtensor<int, 3>::from_shape(shp_as_vec);
            std::vector<std::size_t> expected_shape = {3, 2, 1};
            EXPECT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), ca.shape().begin()));
            EXPECT_TRUE(std::equal(shp.begin(), shp.end(), cb.shape().begin()));
        }
    }

    TEST(xtensor, strided_constructor)
    {
        central_major_result<container_type> cmr;
        xtensor_dynamic cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xtensor, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<container_type> rm;
            int value = 2;
            xtensor_dynamic ra(rm.m_shape, value, layout_type::row_major);
            compare_shape(ra, rm);
            xtensor_dynamic::container_type vec(ra.size(), value);
            EXPECT_EQ(ra.data(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<container_type> cm;
            int value = 2;
            xtensor_dynamic ca(cm.m_shape, value, layout_type::column_major);
            compare_shape(ca, cm);
            xtensor_dynamic::container_type vec(ca.size(), value);
            EXPECT_EQ(ca.data(), vec);
        }
    }

    TEST(xtensor, strided_valued_constructor)
    {
        central_major_result<container_type> cmr;
        int value = 2;
        xtensor_dynamic cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        xtensor_dynamic::container_type vec(cma.size(), value);
        EXPECT_EQ(cma.data(), vec);
    }

    TEST(xtensor, xscalar_constructor)
    {
        xscalar<int> xs(2);
        xtensor<int, 0> a(xs);
        EXPECT_EQ(a(), xs());
    }

    TEST(xtensor, copy_semantic)
    {
        central_major_result<container_type> res;
        int value = 2;
        xtensor_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            xtensor_dynamic b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<container_type> r;
            xtensor_dynamic c(r.m_shape, 0);
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
        xtensor_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            xtensor_dynamic tmp(a);
            xtensor_dynamic b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<container_type> r;
            xtensor_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            xtensor_dynamic tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xtensor, resize)
    {
        xtensor_dynamic a;
        test_resize<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, reshape)
    {
        xtensor_dynamic a;
        test_reshape<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, transpose)
    {
        xtensor_dynamic a;
        test_transpose<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, access)
    {
        xtensor_dynamic a;
        test_access<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, at)
    {
        xtensor_dynamic a;
        test_at<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, element)
    {
        xtensor_dynamic a;
        test_element<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, indexed_access)
    {
        xtensor_dynamic a;
        test_indexed_access<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, broadcast_shape)
    {
        xtensor<int, 4> a;
        test_broadcast(a);
    }

    TEST(xtensor, iterator)
    {
        using xtensor_rm = xtensor<int, 3, layout_type::row_major>;
        using xtensor_cm = xtensor<int, 3, layout_type::column_major>;
        xtensor_rm arm;
        xtensor_cm acm;
        test_iterator<xtensor_rm, xtensor_cm, container_type>(arm, acm);
    }

    TEST(xtensor, zerod)
    {
        xtensor_dynamic a;
        EXPECT_EQ(0, a());
    }

    TEST(xtensor, xiterator)
    {
        xtensor_dynamic a;
        test_xiterator<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, reverse_xiterator)
    {
        xtensor_dynamic a;
        test_reverse_xiterator<xtensor_dynamic, container_type>(a);
    }

    TEST(xtensor, single_element)
    {
        xtensor<int, 1> a = {1};
        xtensor<int, 1> res = 2 * a;
        EXPECT_EQ(2, res(0));
        EXPECT_EQ(2, res(1));

        xtensor<int, 1> b = { 1, 2 };
        xt::xarray<double> ca = a * b;
    }

}
