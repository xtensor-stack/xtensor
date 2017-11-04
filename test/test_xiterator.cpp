/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
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
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());
        size_type nb_inc = shape.back() * shape[shape.size() - 2] + 1;
        int expected = a(1, 0, 1);

        auto iter = a.template begin<layout_type::row_major>();
        auto iter2 = a.template begin<layout_type::row_major>();
        for (size_type i = 0; i < nb_inc; ++i)
        {
            ++iter;
            iter2++;
        }

        EXPECT_EQ(*iter, expected) << "preincrement operator doesn't give expected result";
        EXPECT_EQ(*iter2, expected) << "postincrement operator doesn't give expected result";
    }

    TEST(xiterator, increment_row_major)
    {
        row_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_column_major)
    {
        column_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_central_major)
    {
        central_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    TEST(xiterator, increment_unit_shape)
    {
        unit_shape_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_increment(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment(rm, rm.shape());
        }
    }

    template <class R>
    void test_end(const R& result)
    {
        using size_type = typename R::size_type;
        using shape_type = typename R::shape_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.data();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        size_type size = a.size();
        auto iter = a.begin();
        auto last = a.end();
        for (size_type i = 0; i < size; ++i)
        {
            ++iter;
        }

        EXPECT_EQ(iter, last) << "iterator doesn't reach the end";
    }

    TEST(xiterator, end_row_major)
    {
        row_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm);
        }
    }

    TEST(xiterator, end_column_major)
    {
        column_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm);
        }
    }

    TEST(xiterator, end_central_major)
    {
        central_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm);
        }
    }

    TEST(xiterator, end_unit_shape)
    {
        unit_shape_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_end(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end(rm);
        }
    }

    template <class R, class S>
    void test_decrement(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using shape_type = typename R::shape_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.data();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());
        size_type nb_inc = shape.back() * shape[shape.size() - 2] + 1;
        int expected = a(1, 1, 2);

        auto iter = a.template rbegin<layout_type::row_major>();
        auto iter2 = a.template rbegin<layout_type::row_major>();
        for (size_type i = 0; i < nb_inc; ++i)
        {
            ++iter;
            iter2++;
        }
        EXPECT_EQ(*iter, expected) << "predecrement operator doesn't give expected result";
        EXPECT_EQ(*iter2, expected) << "postdecrement operator doesn't give expected result";
    }

    TEST(xiterator, decrement_row_major)
    {
        row_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_decrement(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement(rm, rm.shape());
        }
    }

    TEST(xiterator, decrement_column_major)
    {
        column_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_decrement(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement(rm, rm.shape());
        }
    }

    TEST(xiterator, decrement_central_major)
    {
        central_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_decrement(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement(rm, rm.shape());
        }
    }

    TEST(xiterator, decrement_unit_shape)
    {
        unit_shape_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_decrement(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement(rm, rm.shape());
        }
    }

    template <class R>
    void test_rend(const R& result)
    {
        using size_type = typename R::size_type;
        using shape_type = typename R::shape_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.data();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        size_type size = a.size();
        auto iter = a.rbegin();
        auto last = a.rend();
        for (size_type i = 0; i < size; ++i)
        {
            ++iter;
        }

        EXPECT_EQ(iter, last) << "reverse iterator doesn't reach the end";
    }

    TEST(xiterator, reverse_end_row_major)
    {
        row_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_rend(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend(rm);
        }
    }

    TEST(xiterator, reverse_end_column_major)
    {
        column_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_rend(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend(rm);
        }
    }

    TEST(xiterator, reverse_end_central_major)
    {
        central_major_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_rend(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend(rm);
        }
    }

    TEST(xiterator, reverse_end_unit_shape)
    {
        unit_shape_result<> rm;
        {
            SCOPED_TRACE("same shape");
            test_rend(rm);
        }

        {
            SCOPED_TRACE("broadcasting shape");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend(rm);
        }
    }

    TEST(xiterator, broadcast)
    {
        EXPECT_TRUE(broadcastable(std::vector<size_t>({3, 2, 1}), std::vector<size_t>({1, 2, 1})));
        EXPECT_TRUE(broadcastable(std::vector<size_t>({3, 2, 1}), std::vector<size_t>({1, 1})));
        EXPECT_TRUE(broadcastable(std::vector<size_t>({1, 1}), std::vector<size_t>({2, 2, 1})));
        EXPECT_FALSE(broadcastable(std::vector<size_t>({3, 2, 1}), std::vector<size_t>({2, 2, 1})));
    }

    TEST(xiterator, pointer)
    {
        xarray<double> m {{3, 4}, {6, 5}};
        constexpr layout_type l = xarray<double>::static_layout == layout_type::column_major ?
            layout_type::row_major : layout_type::column_major;
        auto it = m.begin<l>();
        EXPECT_EQ(*(it.operator->()), 3);
    }

    TEST(xiterator, cross_layout)
    {
        xarray<int, layout_type::row_major> a = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        xarray<int, layout_type::column_major> b = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

        // Performs an element-wise comparison via iterators and ensures the default traversal
        // of a container is consistent for any layout.
        bool res = (a == b);
        EXPECT_TRUE(res);
    }
}
