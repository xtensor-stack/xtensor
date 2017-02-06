/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

    TEST(xbuilder, ones)
    {
        auto m = ones<double>({1, 2});
        ASSERT_EQ(2, m.dimension());
        ASSERT_EQ(1.0, m(0, 1));
        xarray<double> m_assigned = m;
        ASSERT_EQ(1.0, m_assigned(0, 1));
    }

    TEST(xbuilder, arange_simple)
    {
        auto ls = arange<double>(50);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 0);
        auto ls_49 = ls(49);
        ASSERT_EQ(49, ls_49);
        ASSERT_EQ(ls(29), 29);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 50);
        ASSERT_EQ(m_assigned[{0}], 0);
        ASSERT_EQ(m_assigned[{49}], 49);
        ASSERT_EQ(m_assigned[{29}], 29);
    }

    TEST(xbuilder, arange_min_max)
    {
        auto ls = arange<unsigned int>(10u, 20u);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {10};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10);
        ASSERT_EQ(ls(9), 19);
        ASSERT_EQ(ls(2), 12);
        xarray<unsigned int> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 10);
        ASSERT_EQ(m_assigned[{0}], 10);
        ASSERT_EQ(m_assigned[{9}], 19);
        ASSERT_EQ(m_assigned[{2}], 12);
    }

    TEST(xbuilder, arange_min_max_step)
    {
        auto ls = arange<float>(10, 20, 0.5f);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {20};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10);
        ASSERT_EQ(ls(10), 15);
        ASSERT_EQ(ls(3), 11.5f);
        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 20);
        ASSERT_EQ(m_assigned[{0}], 10);
        ASSERT_EQ(m_assigned(10), 15);
        ASSERT_EQ(m_assigned(3), 11.5f);

        auto l3 = arange<float>(0, 1, 0.3f);
        decltype(l3)::shape_type expected_shape_2 = {4};
        ASSERT_EQ(l3.shape(), expected_shape_2);
        ASSERT_EQ(l3[{0}], 0);
        ASSERT_EQ(3.f * 0.3f, l3[{3}]);
    }

    TEST(xbuilder, linspace)
    {
        auto ls = linspace<float>(20.f, 50.f);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);
        ASSERT_EQ(ls(49), 50.f);

        float at_3 = 20 + 3 * (50.f - 20.f) / (50.f - 1.f);
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 50);
        ASSERT_EQ(m_assigned[{0}], 20);
        ASSERT_EQ(m_assigned(49), 50);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, linspace_n_samples_endpoint)
    {
        auto ls = linspace<float>(20.f, 50.f, 100, false);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {100};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);

        float at_end = 49.7f;
        ASSERT_EQ(ls(99), at_end);

        float at_3 = 20.9f;
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 100);
        ASSERT_EQ(m_assigned[{0}], 20);
        ASSERT_EQ(m_assigned(99), at_end);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, logspace)
    {
        auto ls = logspace<double>(2., 3., 4);
        ASSERT_EQ(ls.dimension(), 1);
        decltype(ls)::shape_type expected_shape = {4};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 100);

        double at_1 = std::pow(10.0, (2.0 + 1.0/3.0));
        ASSERT_EQ(ls(1), at_1);

        ASSERT_EQ(ls(3), 1000);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 4);
        ASSERT_EQ(m_assigned[{0}], 100);
        ASSERT_EQ(m_assigned(1), at_1);
        ASSERT_EQ(m_assigned(3), 1000);
    }

    TEST(xbuilder, eye)
    {
        auto e = eye(5);
        ASSERT_EQ(2, e.dimension());
        decltype(e)::shape_type expected_shape = {5, 5};
        ASSERT_EQ(expected_shape, e.shape());

        ASSERT_EQ(true, e(1, 1));
        xindex idx({1, 0});
        ASSERT_EQ(false, e[idx]);

        xarray<bool> m_assigned = e;
        ASSERT_EQ(true, m_assigned(2, 2));
        ASSERT_EQ(false, m_assigned(4, 2));

        xt::xindex idx2({2, 2});
        ASSERT_EQ(true, e.element(idx2.begin(), idx2.end()));
    }
}
