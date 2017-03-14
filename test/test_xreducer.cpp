/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xreducer.hpp"

namespace xt
{
    struct xreducer_features
    {
        using axes_type = std::array<std::size_t, 2>;
        axes_type m_axes;
        xarray<double> m_a;
        using shape_type = xarray<double>::shape_type;

        using func = std::plus<double>;
        xreducer<func, const xarray<double>&, axes_type> m_red;

        xreducer_features();
    };

    xreducer_features::xreducer_features()
        : m_axes({ 1, 3 }), m_a(ones<double>({3, 2, 4, 6, 5})),
          m_red(func(), m_a, m_axes)
    {
        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 6; ++j)
            {
                m_a(1, i, 1, j, 1) = 2;
            }
        }
    }

    TEST(xreducer, shape)
    {
        xreducer_features features;
        xreducer_features::shape_type s = { 3, 4, 5 };
        EXPECT_EQ(s, features.m_red.shape());
    }

    TEST(xreducer, access)
    {
        xreducer_features features;
        EXPECT_EQ(12, features.m_red(0, 0, 0));
        EXPECT_EQ(24, features.m_red(1, 1, 1));
    }

    TEST(xreducer, iterator)
    {
        xreducer_features features;
        auto iter = features.m_red.cbegin();
        auto iter_end = features.m_red.cend();
        const xreducer_features::shape_type& s = features.m_red.shape();
        std::size_t nb_iter = 1;
        nb_iter = std::accumulate(s.cbegin(), s.cend(), nb_iter, std::multiplies<std::size_t>());
        std::advance(iter, nb_iter);
        EXPECT_EQ(iter_end, iter);
    }

    TEST(xreducer, assign)
    {
        xreducer_features features;
        xarray<double> res = features.m_red;
        xarray<double> expected = 12 * ones<double>({ 3, 4, 5 });
        expected(1, 1, 1) = 24;
        EXPECT_EQ(expected, res);
    }

    TEST(xreducer, sum)
    {
        xreducer_features features;
        xarray<double> res = sum(features.m_a, features.m_axes);
        xarray<double> expected = 12 * ones<double>({ 3, 4, 5 });
        expected(1, 1, 1) = 24;
        EXPECT_EQ(expected, res);
    }
}