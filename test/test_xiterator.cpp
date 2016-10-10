#include "gtest/gtest.h"
#include "xarray/xarray.hpp"
#include "test_common.hpp"

namespace qs
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

        ASSERT_EQ(*iter, expected) << "preincrement operator doesn't give expected result";
        ASSERT_EQ(*iter2, expected) << "postincrement operator doesn't give expected result";
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

        ASSERT_EQ(iter, last) << "iterator doesn't reach the end";
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

