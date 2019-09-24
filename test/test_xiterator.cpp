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
#include "xtensor/xtensor.hpp"

#include "test_common.hpp"

namespace xt
{
    template <class R>
    class xiterator_test : public ::testing::Test
    {
    public:

        using result_type = R;
    };

    using testing_types = ::testing::Types<row_major_result<>, column_major_result<>,
        central_major_result<>, unit_shape_result<>>;
    TYPED_TEST_CASE(xiterator_test, testing_types);

    using std::size_t;

    template <layout_type L, class R, class S>
    void test_increment(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        auto offset = shape.size() - a.dimension();
        auto broadcasting_stride = std::accumulate(shape.cbegin(), shape.cbegin() + offset, size_type(1), std::multiplies<size_type>());
        size_type nb_inc = (L == layout_type::row_major ?
            shape.back() * shape[shape.size() - 2] + 1 :
            broadcasting_stride * (result.shape().front() * result.shape()[1] + 1));

        int expected = a(1, 0, 1);

        auto iter = a.template begin<L>(shape);
        auto iter2 = a.template begin<L>(shape);
        for (size_type i = 0; i < nb_inc; ++i)
        {
            ++iter;
            iter2++;
        }

        EXPECT_EQ(*iter, expected) << "preincrement operator doesn't give expected result";
        EXPECT_EQ(*iter2, expected) << "postincrement operator doesn't give expected result";
    }

    TYPED_TEST(xiterator_test, increment)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_increment<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_increment<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_increment<layout_type::column_major>(rm, rm.shape());
        }
    }

    template <layout_type L, class R, class S>
    void test_random_increment(const R& result, const S& shape)
    {
        using difference_type = typename R::difference_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        auto offset = shape.size() - a.dimension();
        auto broadcasting_stride = std::accumulate(shape.cbegin(), shape.cbegin() + offset, difference_type(1), std::multiplies<difference_type>());
        auto nb_inc = L == layout_type::row_major ?
            difference_type(shape.back() * shape[shape.size() - 2] + 1) :
            broadcasting_stride * difference_type(result.shape().front() * result.shape()[1] + 1);

        int expected = a(1, 0, 1);

        auto iter = a.template begin<L>();
        auto iter2 = a.template begin<L>();

        iter += nb_inc;
        auto iter3 = iter2 + nb_inc;

        EXPECT_EQ(*iter, expected) << "preincrement operator doesn't give expected result";
        EXPECT_EQ(*iter3, expected) << "postincrement operator doesn't give expected result";
        EXPECT_EQ(iter2[nb_inc], expected) << "postincrement operator doesn't give expected result";
    }

    TYPED_TEST(xiterator_test, random_increment)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_random_increment<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_random_increment<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_random_increment<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_random_increment<layout_type::column_major>(rm, rm.shape());
        }
    }

    template <layout_type L, class R, class S>
    void test_end(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        size_type size = compute_size(shape);
        auto iter = a.template begin<L>(shape);
        auto last = a.template end<L>(shape);
        for (size_type i = 0; i < size; ++i)
        {
            ++iter;
        }

        EXPECT_EQ(iter, last) << "iterator doesn't reach the end";
        EXPECT_FALSE(iter < last);
    }

    TYPED_TEST(xiterator_test, end)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_end<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end<layout_type::row_major>(rm, sh);
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_end<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_end<layout_type::column_major>(rm, sh);
        }
    }

    template <layout_type L, class R, class S>
    void test_decrement(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        auto offset = shape.size() - a.dimension();
        auto broadcasting_stride = std::accumulate(shape.cbegin(), shape.cbegin() + offset, size_type(1), std::multiplies<size_type>());
        size_type nb_inc = (L == layout_type::row_major ?
            shape.back() * shape[shape.size() - 2] + 1 :
            broadcasting_stride * (result.shape().front() * result.shape()[1] + 1));

        int expected = a(1, 1, 2);

        auto iter = a.template rbegin<L>(shape);
        auto iter2 = a.template rbegin<L>(shape);
        for (size_type i = 0; i < nb_inc; ++i)
        {
            ++iter;
            iter2++;
        }
        EXPECT_EQ(*iter, expected) << "predecrement operator doesn't give expected result";
        EXPECT_EQ(*iter2, expected) << "postdecrement operator doesn't give expected result";
    }

    TYPED_TEST(xiterator_test, decrement)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_decrement<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement<layout_type::row_major>(rm, sh);
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_decrement<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_decrement<layout_type::column_major>(rm, sh);
        }
    }

    template <layout_type L, class R, class S>
    void test_random_decrement(const R& result, const S& shape)
    {
        using difference_type = typename R::difference_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        auto offset = shape.size() - a.dimension();
        auto broadcasting_stride = std::accumulate(shape.cbegin(), shape.cbegin() + offset, difference_type(1), std::multiplies<difference_type>());
        
        auto nb_inc = L == layout_type::row_major ?
            difference_type(shape.back() * shape[shape.size() - 2] + 1) :
            broadcasting_stride * difference_type(result.shape().front() * result.shape()[1] + 1);

        int expected = a(1, 1, 2);

        auto iter = a.template rbegin<L>(shape);
        auto iter2 = a.template rbegin<L>(shape);

        iter += nb_inc;
        auto iter3 = iter2 + nb_inc;

        EXPECT_EQ(*iter, expected) << "predecrement operator doesn't give expected result";
        EXPECT_EQ(*iter3, expected) << "postdecrement operator doesn't give expected result";
    }

    TYPED_TEST(xiterator_test, random_decrement)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_random_decrement<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_random_decrement<layout_type::row_major>(rm, sh);
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_random_decrement<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_random_decrement<layout_type::column_major>(rm, sh);
        }
    }

    template <layout_type L, class R, class S>
    void test_rend(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());

        size_type size = compute_size(shape);
        auto iter = a.template rbegin<L>(shape);
        auto last = a.template rend<L>(shape);

        EXPECT_EQ(*iter, data.back()) << "dereferencing rbegin does not result in last element";
        EXPECT_EQ(*last, data.front()) << "dereferencing rend does not result in first element";

        for (size_type i = 0; i < size; ++i)
        {
            ++iter;
        }

        EXPECT_EQ(iter, last) << "reverse iterator doesn't reach the end";
    }

    TYPED_TEST(xiterator_test, reverse_end)
    {
        typename TestFixture::result_type rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_rend<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend<layout_type::row_major>(rm, sh);
        }

        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_rend<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_rend<layout_type::column_major>(rm, sh);
        }
    }

    template <layout_type L, class R, class S>
    void test_minus(const R& result, const S& shape)
    {
        using size_type = typename R::size_type;
        using difference_type = typename R::difference_type;
        using vector_type = typename R::vector_type;
        vector_type data = result.storage();
        xarray_adaptor<typename R::vector_type, layout_type::dynamic> a(data, result.shape(), result.strides());
        
        size_type size = shape.size();
        difference_type nb_inc = difference_type(L == layout_type::row_major ?
            shape.back() * shape[size - 2] + shape.back() + 2 :
            shape[size - 3] * shape[size - 2] * 2 + shape[size - 3] + 1);
        difference_type nb_inc2 = difference_type(L == layout_type::row_major ?
            shape.back() * shape[size - 2] * 2  + 3 :
            shape[size - 3] * shape[size - 2] * 3 + 2);

        ptrdiff_t expected = ptrdiff_t(nb_inc2 - nb_inc);

        auto iter = a.template begin<L>() + nb_inc;
        auto iter2 = a.template begin<L>() + nb_inc2;
        EXPECT_EQ(iter2 - iter, expected) << "operator- doesn't give expected result";

        auto riter = a.template rbegin<L>() + nb_inc;
        auto riter2 = a.template rbegin<L>() + nb_inc2;
        EXPECT_EQ(riter2 - riter, expected) << "operator- doesn't give expected result";

        auto diff = a.template end<L>() - a.template begin<L>();
        EXPECT_EQ(size_type(diff), a.size());

        auto rdiff = a.template rend<L>() - a.template rbegin<L>();
        EXPECT_EQ(size_type(rdiff), a.size());
    }

    TEST(xiterator, row_major_minus)
    {
        row_major_result<> rm;
        {
            SCOPED_TRACE("same shape - row_major iterator");
            test_minus<layout_type::row_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - row_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_minus<layout_type::row_major>(rm, rm.shape());
        }
    }

    TEST(xiterator, column_major_minus)
    {
        column_major_result<> rm;
        {
            SCOPED_TRACE("same shape - column_major iterator");
            test_minus<layout_type::column_major>(rm, rm.shape());
        }

        {
            SCOPED_TRACE("broadcasting shape - column_major iterator");
            layout_result<>::shape_type sh = rm.shape();
            sh.insert(sh.begin(), 2);
            sh.insert(sh.begin(), 4);
            test_minus<layout_type::column_major>(rm, rm.shape());
        }
    }

    TEST(xiterator, broadcast)
    {
        EXPECT_TRUE(broadcastable(std::vector<size_t>({1, 2, 1}), std::vector<size_t>({ 3, 2, 1 })));
        EXPECT_TRUE(broadcastable(std::vector<size_t>({1, 1}), std::vector<size_t>({ 3, 2, 1 })));
        EXPECT_TRUE(broadcastable(std::vector<size_t>({1, 1}), std::vector<size_t>({2, 2, 1})));
        EXPECT_FALSE(broadcastable(std::vector<size_t>({2, 2, 1}), std::vector<size_t>({ 3, 2, 1 })));
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

    TEST(xiterator, comparisons)
    {
        row_major_result<> rm;
        xarray<int> arr(rm.shape());

        auto rm_iter = arr.template cbegin<layout_type::row_major>();
        auto rm_iter2 = rm_iter + 2;

        EXPECT_TRUE(rm_iter != rm_iter2);
        EXPECT_FALSE(rm_iter == rm_iter2);
        EXPECT_TRUE(rm_iter < rm_iter2);
        EXPECT_TRUE(rm_iter <= rm_iter2);
        EXPECT_FALSE(rm_iter > rm_iter2);
        EXPECT_FALSE(rm_iter >= rm_iter2);

        auto cm_iter = arr.template cbegin<layout_type::column_major>();
        auto cm_iter2 = cm_iter + 2;

        EXPECT_TRUE(cm_iter != cm_iter2);
        EXPECT_FALSE(cm_iter == cm_iter2);
        EXPECT_TRUE(cm_iter < cm_iter2);
        EXPECT_TRUE(cm_iter <= cm_iter2);
        EXPECT_FALSE(cm_iter > cm_iter2);
        EXPECT_FALSE(cm_iter >= cm_iter2);
    }

    TEST(xiterator, assign)
    {
        row_major_result<> rm;
        using vector_type = row_major_result<>::vector_type;
        xarray_adaptor<vector_type, layout_type::dynamic> a(rm.storage(), rm.shape(), rm.strides());
        
        {
            SCOPED_TRACE("row_major iterator");
            xarray<vector_type::value_type> dst(a.shape(), 1);
            std::copy(a.cbegin<layout_type::row_major>(), a.cend<layout_type::row_major>(), dst.begin<layout_type::row_major>());
            EXPECT_EQ(a, dst);
        }

        {
            SCOPED_TRACE("column_major iterator");
            xarray<vector_type::value_type> dst(a.shape(), 1);
            std::copy(a.cbegin<layout_type::column_major>(), a.cend<layout_type::column_major>(), dst.begin<layout_type::column_major>());
            EXPECT_EQ(a, dst);
        }
    }

    TEST(xiterator, revert_assign)
    {
        row_major_result<> rm;
        using vector_type = row_major_result<>::vector_type;
        xarray_adaptor<vector_type, layout_type::dynamic> a(rm.storage(), rm.shape(), rm.strides());

        {
            SCOPED_TRACE("row_major iterator");
            xarray<vector_type::value_type> dst(a.shape(), 1);
            std::copy(a.crbegin<layout_type::row_major>(), a.crend<layout_type::row_major>(), dst.rbegin<layout_type::row_major>());
            EXPECT_EQ(a, dst);
        }

        {
            SCOPED_TRACE("column_major iterator");
            xarray<vector_type::value_type> dst(a.shape(), 1);
            std::copy(a.crbegin<layout_type::column_major>(), a.crend<layout_type::column_major>(), dst.rbegin<layout_type::column_major>());
            EXPECT_EQ(a, dst);
        }
    }

    TEST(xiterator, reverse_corner_cases)
    {
        xtensor<double, 3> a = {{{13, 16},
                                 {13, 16},
                                 {13, 16}}};

        constexpr auto cm = layout_type::column_major;
        constexpr auto rm = layout_type::row_major;
        {
            std::vector<double> exp_iter = {16, 16, 16, 13, 13, 13};
            auto e_iter = exp_iter.begin();
            for (auto it = a.template rbegin<cm>(); it != a.template rend<cm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
        }

        {
            std::vector<double> exp_iter = {16, 13, 16, 13, 16, 13};
            auto e_iter = exp_iter.begin();
            for (auto it = a.template rbegin<rm>(); it != a.template rend<rm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
        }

        {
            std::vector<double> exp_iter = {13, 16, 13, 16, 13, 16};
            auto e_iter = exp_iter.begin();
            for (auto it = a.template begin<rm>(); it != a.template end<rm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
        }

        {
            std::vector<double> exp_iter = {13, 13, 13, 16, 16, 16};
            auto e_iter = exp_iter.begin();
            for (auto it = a.template begin<cm>(); it != a.template end<cm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
        }

        xtensor<double, 3> b = {{{1},
                                 {1}},
                                {{2},
                                 {2}},
                                {{3},
                                 {3}}};

       {
            std::vector<double> exp_iter = {3, 2, 1, 3, 2, 1};
            auto e_iter = exp_iter.begin();
            for (auto it = b.template rbegin<cm>(); it != b.template rend<cm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
       }

       {
            std::vector<double> exp_iter = {3, 3, 2, 2, 1, 1};
            auto e_iter = exp_iter.begin();
            for (auto it = b.template rbegin<rm>(); it != b.template rend<rm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
       }

       {
            std::vector<double> exp_iter = {1, 1, 2, 2, 3, 3};
            auto e_iter = exp_iter.begin();
            for (auto it = b.template begin<rm>(); it != b.template end<rm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
       }

       {
            std::vector<double> exp_iter = {1, 2, 3, 1, 2, 3};
            auto e_iter = exp_iter.begin();
            for (auto it = b.template begin<cm>(); it != b.template end<cm>(); ++it)
            {
                EXPECT_EQ(*e_iter, *it);
                ++e_iter;
            }
            EXPECT_TRUE(e_iter == exp_iter.end());
       }
    }
}
