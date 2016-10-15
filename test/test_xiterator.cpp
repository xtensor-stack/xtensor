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
    using std::size_t;

    template <class R, class S>
    void test_increment(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using shape_type = typename R::shape_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.data();
        xarray_adaptor<typename R::vector_type> a(data, result.shape(), result.strides());
        size_type nb_inc = shape.back() * shape[shape.size() - 2] + 1;
        int expected = a(1, 0, 1);
        
        auto iter = a.begin();
        auto iter2 = a.begin();
        for(size_type i = 0; i < nb_inc; ++i)
        {
            ++iter;
            iter2++;
        }

        EXPECT_EQ(*iter, expected) << "preincrement operator doesn't give expected result";
        EXPECT_EQ(*iter2, expected) << "postincrement operator doesn't give expected result";
    }

    TEST(xiterator, increment_row_major)
    {
        row_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_column_major)
    {
        column_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_central_major)
    {
        central_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_unit_shape)
    {
        unit_shape_result rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    template <class R, class S>
    void test_end(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using shape_type = typename R::shape_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.data();
        xarray_adaptor<typename R::vector_type> a(data, result.shape(), result.strides());

        size_type size = a.size();
        auto iter = a.begin();
        auto last = a.end();
        for(size_type i = 0; i < size; ++i)
        {
            ++iter;
        }

        EXPECT_EQ(iter, last) << "iterator doesn't reach the end";
    }

    TEST(xiterator, end_row_major)
    {
        row_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm, rm.shape());
        }
    }

    TEST(xiterator, end_column_major)
    {
        column_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm, rm.shape());
        }
    }

    TEST(xiterator, end_central_major)
    {
        central_major_result rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm, rm.shape());
        }
    }

    TEST(xiterator, end_unit_shape)
    {
        unit_shape_result rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm, rm.shape());
        }
    }

}

