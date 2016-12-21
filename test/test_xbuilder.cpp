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
#include "xtensor/xio.hpp"

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<size_t>;

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
        shape_t expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 0);
        ASSERT_EQ(ls(49), 49);
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
        auto ls = arange<uint>(10, 20);
        ASSERT_EQ(ls.dimension(), 1);
        shape_t expected_shape = {10};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10);
        ASSERT_EQ(ls(9), 19);
        ASSERT_EQ(ls(2), 12);
        xarray<uint> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 10);
        ASSERT_EQ(m_assigned[{0}], 10);
        ASSERT_EQ(m_assigned[{9}], 19);
        ASSERT_EQ(m_assigned[{2}], 12);
    }

    TEST(xbuilder, arange_min_max_step)
    {
        auto ls = arange<float>(10, 20, 0.5);
        ASSERT_EQ(ls.dimension(), 1);
        shape_t expected_shape = {20};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10);
        ASSERT_EQ(ls(10), 15);
        ASSERT_EQ(ls(3), 11.5);
        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 20);
        ASSERT_EQ(m_assigned[{0}], 10);
        ASSERT_EQ(m_assigned(10), 15);
        ASSERT_EQ(m_assigned(3), 11.5);
    }

    TEST(xbuilder, linspace)
    {
        auto ls = linspace<float>(20, 50);
        ASSERT_EQ(ls.dimension(), 1);
        shape_t expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20);
        ASSERT_EQ(ls(49), 50);

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
        auto ls = linspace<float>(20, 50, 100, false);
        ASSERT_EQ(ls.dimension(), 1);
        shape_t expected_shape = {100};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20);

        float at_end = 49.7;
        ASSERT_EQ(ls(99), at_end);
        
        float at_3 = 20.9;
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
        auto ls = logspace<float>(2, 3, 4);
        ASSERT_EQ(ls.dimension(), 1);
        shape_t expected_shape = {4};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 100);

        float at_1 = std::pow(10.f, (2 + 1.f/3.f));
        ASSERT_EQ(ls(1), at_1);
        
        ASSERT_EQ(ls(3), 1000);
        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), 1);
        ASSERT_EQ(m_assigned.shape()[0], 4);
        ASSERT_EQ(m_assigned[{0}], 100);
        ASSERT_EQ(m_assigned(1), at_1);
        ASSERT_EQ(m_assigned(3), 1000);
    }

    TEST(xbuilder, meshgrid)
    {
        auto x = arange<double>(-5, 5, 0.1);
        auto y = arange<double>(-5, 5, 0.1);
        auto mg = meshgrid(x, y);
        auto xx = std::get<0>(mg);
        auto yy = std::get<1>(mg);

        ASSERT_EQ(xx(3, 3), x(3));
        ASSERT_EQ(yy(10, 5), yy(10, 6));
        
        auto z = sin(pow(xx, 2) + pow(yy, 2)) / (pow(xx, 2) + pow(yy, 2));
        double at_3_3 = 0.0044458453690595342;
        double at_5_4 = -0.013017116600771921;
        double z_3_3 = z[{3, 3}];

        ASSERT_NEAR(z_3_3, at_3_3, 10e-15);
        ASSERT_NEAR(z(5, 4), at_5_4, 10e-15);
    }

    TEST(xbuilder, hstack_vstack)
    {
        xarray<double> a = {{1,2,3}, {4,5,6}, {7,8,9}};
        xarray<uint> b = a * 5;

        auto hstacked = hstack(a, b);
        auto vstacked = vstack(a, b);
        
        shape_t h_expected_shape = {3, 6};
        shape_t v_expected_shape = {6, 3};
        ASSERT_EQ(hstacked.shape(), h_expected_shape);        
        ASSERT_EQ(vstacked.shape(), v_expected_shape);        

        ASSERT_EQ(hstacked(2, 2), 9);
        ASSERT_EQ(hstacked(1, 4), 25);

        double hs_1_4 = hstacked[{1, 4}];
        ASSERT_EQ(hs_1_4, 25);
        
        ASSERT_EQ(vstacked(2, 2), 9);
        ASSERT_EQ(vstacked(5, 2), 45);
        
        double vs_5_2 = vstacked[{5, 2}];
        ASSERT_EQ(vs_5_2, 45);
    }

    TEST(xbuilder, selfstack_onedim)
    {
        xarray<double> a = {1, 2, 3};

        auto vstacked = vstack(a, a);
        shape_t v_expected_shape = {2, 3};
        ASSERT_EQ(vstacked.shape(), v_expected_shape);

        auto hstacked = hstack(a, a);
        shape_t h_expected_shape = {6};
        ASSERT_EQ(hstacked.shape(), h_expected_shape);
        ASSERT_EQ(hstacked(2), hstacked(5));
    }

    TEST(xbuilder, stack_onedim_twodim)
    {
        xarray<double> a = {1, 2, 3};
        xarray<double> b = {{4, 5, 6}, {7, 8, 9}};

        auto vstacked = vstack(a, b);
        std::cout << vstacked << std::endl;
        shape_t v_expected_shape = {3, 3};
        ASSERT_EQ(vstacked.shape(), v_expected_shape);
        ASSERT_EQ(vstacked(2, 2), 9);

        auto vstacked_other = vstack(b, a);
        std::cout << vstacked_other << std::endl;
        ASSERT_EQ(vstacked_other.shape(), v_expected_shape);
        ASSERT_EQ(vstacked_other(2, 2), 3);

        auto hstacked_other = hstack(b, b);
        std::cout << hstacked_other << std::endl;
        shape_t h_expected_shape = {2, 6};
        ASSERT_EQ(hstacked_other.shape(), h_expected_shape);
        ASSERT_EQ(hstacked_other(1, 1), hstacked_other(1, 4));
    }

    TEST(xbuilder, concatenate)
    {
        xarray<double> a = arange<double>(12);
        a.reshape({2, 2, 3});

        auto c = concatenate(a, a, 2);

        shape_t expected_shape = {2, 2, 6};

        ASSERT_EQ(c(1, 1, 2), c(1, 1, 5));
        ASSERT_EQ(c(1, 1, 2), 11);
        ASSERT_EQ(c(1, 1, 5), 11);
    }
}
