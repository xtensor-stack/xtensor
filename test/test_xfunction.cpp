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

    struct xfunction_features
    {
        xarray<int> m_a; // shape = { 3, 2, 4 }
        xarray<int> m_b; // shape = { 3, 1, 4 }
        xarray<int> m_c; // shape = { 4, 3, 2, 4 }

        xfunction_features();
    };

    xfunction_features::xfunction_features()
    {
        row_major_result rm;
        m_a.reshape(rm.shape(), rm.strides());
        std::copy(rm.data().begin(), rm.data().end(), m_a.storage_begin());

        unit_shape_result us;
        m_b.reshape(us.shape(), us.strides());
        std::copy(us.data().begin(), us.data().end(), m_b.storage_begin());

        using shape_type = layout_result::shape_type;
        shape_type sh = { 4, 3, 2, 4};
        m_c.reshape(sh);

        for(size_t i = 0; i < sh[0]; ++i)
            for(size_t j = 0; j < sh[1]; ++j)
                for(size_t k = 0; k < sh[2]; ++k)
                    for(size_t l = 0; l < sh[3]; ++l)
                        m_c(i, j, k, l) = m_a(j, k, l) + static_cast<int>(i);
    }

    TEST(xfunction, broadcast_shape)
    {
        using shape_type = layout_result::shape_type;
        xfunction_features f;

        {
            SCOPED_TRACE("same shape");
            shape_type sh(3, size_t(1));
            bool trivial = (f.m_a + f.m_a).broadcast_shape(sh);
            EXPECT_EQ(sh, f.m_a.shape());
            ASSERT_TRUE(trivial);
        }

        {
            SCOPED_TRACE("different shape");
            shape_type sh(3, size_t(1));
            bool trivial = (f.m_a + f.m_b).broadcast_shape(sh);
            EXPECT_EQ(sh, f.m_a.shape());
            ASSERT_FALSE(trivial);
        }

        {
            SCOPED_TRACE("different dimensions");
            shape_type sh(4, size_t(1));
            bool trivial = (f.m_a + f.m_c).broadcast_shape(sh);
            EXPECT_EQ(sh, f.m_c.shape());
            ASSERT_FALSE(trivial);
        }
    }

    TEST(xfunction, access)
    {
        xfunction_features f;
        size_t i = f.m_a.shape()[0] - 1;
        size_t j = f.m_a.shape()[1] - 1;
        size_t k = f.m_a.shape()[2] - 1;
        
        {
            SCOPED_TRACE("same shape");
            int a = (f.m_a + f.m_a)(i, j, k);
            int b = f.m_a(i, j, k) + f.m_a(i, j, k);
            EXPECT_EQ(a, b);
        }

        {
            SCOPED_TRACE("different shape");
            int a = (f.m_a + f.m_b)(i, j, k);
            int b = f.m_a(i, j, k) + f.m_b(i, 0, k);
            EXPECT_EQ(a, b);
        }

        {
            SCOPED_TRACE("different dimensions");
            int a = (f.m_a + f.m_c)(1, i, j, k);
            int b = f.m_a(i, j, k) + f.m_c(1, i, j, k);
            EXPECT_EQ(a, b);
        }
    }

    void test_xfunction_iterator(const xarray<int>& a, const xarray<int>& b)
    {
        auto func = (a + b);
        auto iter = func.begin();
        auto itera = a.begin();
        auto iterb = b.xbegin(a.shape());
        auto nb_iter = a.shape().back() * 2 + 1;
        for(size_t i = 0; i < nb_iter; ++i)
        {
            ++iter, ++itera, ++iterb;
        }
        EXPECT_EQ(*iter, *itera + *iterb);
    }

    TEST(xfunction, iterator)
    {
        xfunction_features f;

        {
            SCOPED_TRACE("same shape");
            test_xfunction_iterator(f.m_a, f.m_a);
        }

        {
            SCOPED_TRACE("different shape");
            test_xfunction_iterator(f.m_a, f.m_b);
        }

        {
            SCOPED_TRACE("different dimensions");
            test_xfunction_iterator(f.m_c, f.m_a);
        }
    }

    void test_xfunction_iterator_end(const xarray<int>& a, const xarray<int>& b)
    {
        auto func = (a + b);
        auto iter = func.begin();
        auto iter_end = func.end();
        auto size = a.size();
        for(size_t i = 0; i < size; ++i)
        {
            ++iter;
        }
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xfunction, iterator_end)
    {
        xfunction_features f;

        {
            SCOPED_TRACE("same shape");
            test_xfunction_iterator_end(f.m_a, f.m_a);
        }

        {
            SCOPED_TRACE("different shape");
            test_xfunction_iterator_end(f.m_a, f.m_b);
        }

        {
            SCOPED_TRACE("different dimensions");
            test_xfunction_iterator_end(f.m_c, f.m_a);
        }
    }
}

