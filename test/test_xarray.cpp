/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "test_common.hpp"

namespace xt
{
    using xarray_dynamic = xarray<int, layout_type::dynamic>;

    TEST(xarray, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            xarray_dynamic ra(rm.m_shape, layout_type::row_major);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            xarray<int, layout_type::column_major> ca(cm.m_shape);
            compare_shape(ca, cm);
        }

        {
            SCOPED_TRACE("from shape");
            std::array<std::size_t, 3> shp = {5, 4, 2};
            std::vector<std::size_t> shp_as_vec = {5, 4, 2};
            auto ca = xarray<int, layout_type::column_major>::from_shape({3, 2, 1});
            auto cb = xarray<int, layout_type::column_major>::from_shape(shp);
            std::vector<std::size_t> expected_shape = {3, 2, 1};
            EXPECT_EQ(expected_shape, ca.shape());
            EXPECT_EQ(shp_as_vec, cb.shape());
        }
    }

    TEST(xarray, strided_constructor)
    {
        central_major_result<> cmr;
        xarray<int, layout_type::dynamic> cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xarray, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<> rm;
            int value = 2;
            xarray_dynamic ra(rm.m_shape, value, layout_type::row_major);
            compare_shape(ra, rm);
            xarray_dynamic::container_type vec(ra.size(), value);
            EXPECT_EQ(ra.data(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<> cm;
            int value = 2;
            xarray<int, layout_type::column_major> ca(cm.m_shape, value);
            compare_shape(ca, cm);
            xarray_dynamic::container_type vec(ca.size(), value);
            EXPECT_EQ(ca.data(), vec);
        }
    }

    TEST(xarray, strided_valued_constructor)
    {
        central_major_result<> cmr;
        int value = 2;
        xarray<int, layout_type::dynamic> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        xarray_dynamic::container_type vec(cma.size(), value);
        EXPECT_EQ(cma.data(), vec);
    }

    TEST(xarray, xscalar_constructor)
    {
        xscalar<int> xs(2);
        xarray<int> a(xs);
        EXPECT_EQ(a(), xs());
    }

    TEST(xarray, copy_semantic)
    {
        central_major_result<> res;
        int value = 2;
        xarray_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            xarray_dynamic b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            xarray_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xarray, move_semantic)
    {
        central_major_result<> res;
        int value = 2;
        xarray_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            xarray_dynamic tmp(a);
            xarray_dynamic b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            xarray_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            xarray_dynamic tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(xarray, resize)
    {
        xarray_dynamic a;
        test_resize(a);
    }

    TEST(xarray, reshape)
    {
        xarray_dynamic a;
        test_reshape(a);
    }

    TEST(xarray, transpose)
    {
        xarray_dynamic a;
        test_transpose(a);
    }

    TEST(xarray, transpose_row)
    {
        xarray<float> a = {{0, 1, 1, 1}};
        xarray<float> res = xt::transpose(a);

        xarray<float>::shape_type sh = {4, 1};
        EXPECT_EQ(res.shape(), sh);
        EXPECT_EQ(res(0, 0), 0.f);
        EXPECT_EQ(res(1, 0), 1.f);
        EXPECT_EQ(res(2, 0), 1.f);
        EXPECT_EQ(res(3, 0), 1.f);
    }

    TEST(xarray, access)
    {
        xarray_dynamic a;
        test_access(a);
    }

    TEST(xarray, at)
    {
        xarray_dynamic a;
        test_at(a);
    }

    TEST(xarray, element)
    {
        xarray_dynamic a;
        test_element(a);
    }

    TEST(xarray, indexed_access)
    {
        xarray_dynamic a;
        test_indexed_access(a);
    }

    TEST(xarray, broadcast_shape)
    {
        xarray_dynamic a;
        test_broadcast(a);
        test_broadcast2(a);
    }

    TEST(xarray, iterator)
    {
        xarray<int, layout_type::row_major> arm;
        xarray<int, layout_type::column_major> acm;
        test_iterator(arm, acm);
    }

    TEST(xarray, initializer_list)
    {
        xarray_dynamic a0(1);
        xarray_dynamic a1({1, 2});
        xarray_dynamic a2({{1, 2}, {2, 4}, {5, 6}});
        EXPECT_EQ(1, a0());
        EXPECT_EQ(2, a1(1));
        EXPECT_EQ(4, a2(1, 1));
    }

    TEST(xarray, zerod)
    {
        xarray_dynamic a;
        EXPECT_EQ(0, a());

        xarray_dynamic b = {1, 2, 3};
        xarray_dynamic c(2 + xt::sum(b));
        EXPECT_EQ(8, c());
    }

    TEST(xarray, xiterator)
    {
        xarray_dynamic a;
        test_xiterator(a);
    }

    TEST(xarray, reverse_xiterator)
    {
        xarray_dynamic a;
        test_reverse_xiterator(a);
    }

    TEST(xarray, cross_layout_assign)
    {
        xarray<int, layout_type::row_major> a = {{1, 2, 3, 4},
                                                 {5, 6, 7, 8}};
        xarray<int, layout_type::column_major> b = {{1, 2, 3, 4},
                                                    {5, 6, 7, 8}};

        xarray<int, layout_type::column_major> ra = a;
        EXPECT_EQ(b, ra);

        xarray<int, layout_type::row_major> rb = b;
        EXPECT_EQ(a, rb);
    }

    TEST(xarray, end_optimized_stride)
    {
        xarray_dynamic a = {1};
        EXPECT_FALSE((a.begin() == a.end()));
    }
}
