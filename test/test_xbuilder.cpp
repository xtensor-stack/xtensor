/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"

#include "xtensor/xio.hpp"
#include <sstream>



namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

    TEST(xbuilder, ones)
    {
        auto m = ones<double>({1, 2});
        ASSERT_EQ(size_t(2), m.dimension());
        ASSERT_EQ(1.0, m(0, 1));
        xarray<double> m_assigned = m;
        ASSERT_EQ(1.0, m_assigned(0, 1));

        // assignment with narrowing type cast
        // (check that the compiler doesn't issue a warning)
        xarray<uint8_t> c = cast<uint8_t>(m);
        ASSERT_EQ(1, c(0, 1));
    }

    TEST(xbuilder, index_placeholder_1D)
    {
        auto m = arange<int>({3});
        auto i0 = index_placeholder<int,1,0>();


        ASSERT_EQ(size_t(1), i0.dimension());
        ASSERT_EQ(size_t(1), i0.shape()[0]);

        auto res  = m + i0;

        ASSERT_EQ(int(0), res(0));
        ASSERT_EQ(int(2), res(1));
        ASSERT_EQ(int(4), res(2));

    }

    TEST(xbuilder, index_placeholder_2D_simple_matching_dim)
    {   
        auto m  = ones<int>({2,3});
        auto i0 = index_placeholder<int,2, 0>();
        auto i1 = index_placeholder<int,2, 1>();

        auto res  = m + i0 + 2*i1;

        ASSERT_EQ(size_t(2), res.shape()[0]);
        ASSERT_EQ(size_t(3), res.shape()[1]);

        ASSERT_EQ(int(1+0+2*0), res(0,0));
        ASSERT_EQ(int(1+0+2*1), res(0,1));
        ASSERT_EQ(int(1+0+2*2), res(0,2));
        ASSERT_EQ(int(1+1+2*0), res(1,0));
        ASSERT_EQ(int(1+1+2*1), res(1,1));
        ASSERT_EQ(int(1+1+2*2), res(1,2));
    }

    TEST(xbuilder, index_placeholder_2D_advanced_matching_dim)
    {
        // broadcasted shape   2,3
        auto a  = 2*ones<int>({1,3});
        auto b  = 3*ones<int>({2,1});
        auto c  = 4*ones<int>({1,1});

        // first index
        auto i0 = index_placeholder<int,2, 0>();

        // second index
        auto i1 = index_placeholder<int,2, 1>();

        // nontrivial expression
        auto res_a  = a + b*(i0+1) + (c  * i1)+2;
        auto res_b  = a + (1+i0)*b + (i1 * c )+2;

        ASSERT_EQ(size_t(2), res_a.shape()[0]);
        ASSERT_EQ(size_t(3), res_a.shape()[1]);

        ASSERT_EQ(size_t(2), res_b.shape()[0]);
        ASSERT_EQ(size_t(3), res_b.shape()[1]);


        ASSERT_EQ(int(2 + 3*(0+1) + 4*0+2), res_a(0,0));
        ASSERT_EQ(int(2 + 3*(0+1) + 4*1+2), res_a(0,1));
        ASSERT_EQ(int(2 + 3*(0+1) + 4*2+2), res_a(0,2));
        ASSERT_EQ(int(2 + 3*(1+1) + 4*0+2), res_a(1,0));
        ASSERT_EQ(int(2 + 3*(1+1) + 4*1+2), res_a(1,1));
        ASSERT_EQ(int(2 + 3*(1+1) + 4*2+2), res_a(1,2));

        ASSERT_EQ(res_a(0,0), res_b(0,0));
        ASSERT_EQ(res_a(0,1), res_b(0,1));
        ASSERT_EQ(res_a(0,2), res_b(0,2));
        ASSERT_EQ(res_a(1,0), res_b(1,0));
        ASSERT_EQ(res_a(1,1), res_b(1,1));
        ASSERT_EQ(res_a(1,2), res_b(1,2));
    }

    TEST(xbuilder, index_placeholder_2D_fancy_non_matching_dim)
    {
        // broadcasted shape   2,3,2
        auto a  = 2*ones<int>({1,3,2});
        auto b  = 3*ones<int>({2,1,1});
        auto c  = 4*ones<int>({1,1,1});

        // 0 and 1 index, both 2d 
        // => will become index 1 and 2 index in 3d after broadcasting
        auto i0 = index_placeholder<int,2, 0>();
        auto i1 = index_placeholder<int,2, 1>();

        // the above index placeholders
        // are the same as these: (after broadcasting)
        auto j1 = index_placeholder<int,3, 1>();
        auto j2 = index_placeholder<int,3, 2>();

        {
            auto res_a = a + 3*i0 + 2*i1;
            auto res_b = a + 3*j1 + 2*j2;



            ASSERT_EQ(2+3*0+2*0, res_a(0,0,0));
            ASSERT_EQ(2+3*0+2*1, res_a(0,0,1));
            ASSERT_EQ(2+3*1+2*0, res_a(0,1,0));
            ASSERT_EQ(2+3*1+2*1, res_a(0,1,1));
            ASSERT_EQ(2+3*2+2*0, res_a(0,2,0));
            ASSERT_EQ(2+3*2+2*1, res_a(0,2,1));
            ASSERT_EQ(2+3*0+2*0, res_a(1,0,0));
            ASSERT_EQ(2+3*0+2*1, res_a(1,0,1));
            ASSERT_EQ(2+3*1+2*0, res_a(1,1,0));
            ASSERT_EQ(2+3*1+2*1, res_a(1,1,1));
            ASSERT_EQ(2+3*2+2*0, res_a(1,2,0));
            ASSERT_EQ(2+3*2+2*1, res_a(1,2,1));
        


            ASSERT_EQ(res_b(0,0,0), res_a(0,0,0));
            ASSERT_EQ(res_b(0,0,1), res_a(0,0,1));
            ASSERT_EQ(res_b(0,1,0), res_a(0,1,0));
            ASSERT_EQ(res_b(0,1,1), res_a(0,1,1));
            ASSERT_EQ(res_b(0,2,0), res_a(0,2,0));
            ASSERT_EQ(res_b(0,2,1), res_a(0,2,1));
            ASSERT_EQ(res_b(1,0,0), res_a(1,0,0));
            ASSERT_EQ(res_b(1,0,1), res_a(1,0,1));
            ASSERT_EQ(res_b(1,1,0), res_a(1,1,0));
            ASSERT_EQ(res_b(1,1,1), res_a(1,1,1));
            ASSERT_EQ(res_b(1,2,0), res_a(1,2,0));
            ASSERT_EQ(res_b(1,2,1), res_a(1,2,1));

        }
    }
    TEST(xbuilder, index_placeholder_blitz_example){

            // Fill a two-dimensional array with a radially
            // symmetric, decaying sinusoid
            const static std::size_t DIM = 2;

            // Create the array
            int N = 64;           
            auto array = ones<double>({N,N});

            // Some parameters
            double midpoint = (N-1)/2.;
            int cycles = 3;
            double omega = 2.0 * M_PI * cycles / double(N);
            double tau = - 10.0 / N;

            // Index placeholders
            auto i = index_placeholder<double, DIM, 0>(); // first index
            auto j = index_placeholder<double, DIM, 1>(); // second index
    
            // Fill the array
            auto data = cos(omega * sqrt(pow(i-midpoint, 2) + pow(j-midpoint, 2)))
                * exp(tau * sqrt(pow(i-midpoint, 2) + pow(j-midpoint, 2)));



            // TESTING
            for(auto x1=0; x1<N; ++x1)
            for(auto x0=0; x0<N; ++x0){

                auto ii = double(x0);
                auto jj = double(x1);

                auto should_val = std::cos(omega * std::sqrt(std::pow(ii-midpoint, 2) + std::pow(jj-midpoint, 2)))
                    * std::exp(tau * std::sqrt(std::pow(ii-midpoint, 2) + std::pow(jj-midpoint, 2)));
                
                auto is_val = data(x0, x1);

                ASSERT_NEAR(is_val, should_val, 0.000001);
            }

    }
    TEST(xbuilder, arange_simple)
    {
        auto ls = arange<double>(50);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 0);
        auto ls_49 = ls(49);
        ASSERT_EQ(49, ls_49);
        ASSERT_EQ(ls(29), 29);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(50));
        ASSERT_EQ(m_assigned[{0}], 0);
        ASSERT_EQ(m_assigned[{49}], 49);
        ASSERT_EQ(m_assigned[{29}], 29);

        xarray<double> b({2, 50}, 1.);
        xarray<double> res = b + ls;
        ASSERT_EQ(50, res(1, 49));
    }

    TEST(xbuilder, arange_min_max)
    {
        auto ls = arange<unsigned int>(10u, 20u);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {10};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10u);
        ASSERT_EQ(ls(9), 19u);
        ASSERT_EQ(ls(2), 12u);
        xarray<unsigned int> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(10));
        ASSERT_EQ(m_assigned[{0}], 10u);
        ASSERT_EQ(m_assigned[{9}], 19u);
        ASSERT_EQ(m_assigned[{2}], 12u);
    }

    TEST(xbuilder, arange_min_max_step)
    {
        auto ls = arange<float>(10, 20, 0.5f);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {20};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10.f);
        ASSERT_EQ(ls(10), 15.f);
        ASSERT_EQ(ls(3), 11.5f);
        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(20));
        ASSERT_EQ(m_assigned[{0}], 10.f);
        ASSERT_EQ(m_assigned(10), 15.f);
        ASSERT_EQ(m_assigned(3), 11.5f);

        auto l3 = arange<float>(0, 1, 0.3f);
        decltype(l3)::shape_type expected_shape_2 = {4};
        ASSERT_EQ(l3.shape(), expected_shape_2);
        ASSERT_EQ(l3[{0}], 0.f);
        ASSERT_EQ(3.f * 0.3f, l3[{3}]);
    }

    TEST(xbuilder, linspace)
    {
        auto ls = linspace<float>(20.f, 50.f);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);
        ASSERT_EQ(ls(49), 50.f);

        float at_3 = 20 + 3 * (50.f - 20.f) / (50.f - 1.f);
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(50));
        ASSERT_EQ(m_assigned[{0}], 20.f);
        ASSERT_EQ(m_assigned(49), 50.f);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, linspace_n_samples_endpoint)
    {
        auto ls = linspace<float>(20.f, 50.f, 100, false);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {100};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);

        float at_end = 49.7f;
        ASSERT_EQ(ls(99), at_end);

        float at_3 = 20.9f;
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(100));
        ASSERT_EQ(m_assigned[{0}], 20.f);
        ASSERT_EQ(m_assigned(99), at_end);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, linspace_integer)
    {
        xarray<int> ls = linspace<int>(0, 10, 13);
        xarray<int> expected = {0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10};

        ASSERT_TRUE(all(equal(ls, expected)));
    }

    TEST(xbuilder, logspace)
    {
        auto ls = logspace<double>(2., 3., 4);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {4};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 100);

        double at_1 = std::pow(10.0, (2.0 + 1.0 / 3.0));
        ASSERT_EQ(ls(1), at_1);

        ASSERT_EQ(ls(3), 1000);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(4));
        ASSERT_EQ(m_assigned[{0}], 100);
        ASSERT_EQ(m_assigned(1), at_1);
        ASSERT_EQ(m_assigned(3), 1000);
    }

    TEST(xbuilder, eye)
    {
        auto e = eye(5);
        ASSERT_EQ(size_t(2), e.dimension());
        decltype(e)::shape_type expected_shape = {5, 5};
        ASSERT_EQ(expected_shape, e.shape());

        ASSERT_TRUE(e(1, 1));
        xindex idx({1, 0});
        ASSERT_FALSE(e[idx]);

        xarray<bool> m_assigned = e;
        ASSERT_TRUE(m_assigned(2, 2));
        ASSERT_FALSE(m_assigned(4, 2));

        xindex idx2({2, 2});
        ASSERT_TRUE(e.element(idx2.begin(), idx2.end()));

        ASSERT_TRUE(e[idx2]);
        ASSERT_TRUE((e[{2, 2}]));
    }

    TEST(xbuilder, concatenate)
    {
        xarray<double> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        auto c = concatenate(xtuple(a, a, a), 2);

        shape_t expected_shape = {2, 2, 9};
        ASSERT_EQ(expected_shape, c.shape());
        ASSERT_EQ(c(1, 1, 2), c(1, 1, 5));
        ASSERT_EQ(11, c(1, 1, 2));
        ASSERT_EQ(11, c(1, 1, 5));

        xarray<double> e = {{1, 2, 3}};
        xarray<double> f = {{2, 3, 4}};
        xarray<double> k = concatenate(xtuple(e, f));
        xarray<double> l = concatenate(xtuple(e, f), 1);

        shape_t ex_k = {2, 3};
        shape_t ex_l = {1, 6};
        ASSERT_EQ(ex_k, k.shape());
        ASSERT_EQ(ex_l, l.shape());
        ASSERT_EQ(4, k(1, 2));
        ASSERT_EQ(3, l(0, 2));
        ASSERT_EQ(3, l(0, 4));

        auto t = concatenate(xtuple(arange(2), arange(2, 5), arange(5, 8)));
        ASSERT_TRUE(arange(8) == t);
    }

    TEST(xbuilder, stack)
    {
        xarray<double> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        auto c = stack(xtuple(a, a, a), 2);

        shape_t expected_shape = {2, 2, 3, 3};

        ASSERT_EQ(expected_shape, c.shape());
        ASSERT_EQ(c(1, 1, 0, 2), c(1, 1, 1, 2));
        ASSERT_EQ(c(1, 1, 0, 2), c(1, 1, 2, 2));
        ASSERT_EQ(11, c(1, 1, 1, 2));
        ASSERT_EQ(11, c(1, 1, 2, 2));

        auto e = arange(1, 4);
        xarray<double> f = {2, 3, 4};
        xarray<double> k = stack(xtuple(e, f));
        xarray<double> l = stack(xtuple(e, f), 1);

        shape_t ex_k = {2, 3};
        shape_t ex_l = {3, 2};
        ASSERT_EQ(ex_k, k.shape());
        ASSERT_EQ(ex_l, l.shape());
        ASSERT_EQ(4, k(1, 2));
        ASSERT_EQ(3, l(1, 1));
        ASSERT_EQ(3, l(2, 0));

        auto t = stack(xtuple(arange(3), arange(3, 6), arange(6, 9)));
        xarray<double> ar = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
        ASSERT_TRUE(t == ar);
    }

    TEST(xbuilder, meshgrid)
    {
        auto mesh = meshgrid(linspace<double>(0.0, 1.0, 3), linspace<double>(0.0, 1.0, 2));
        xarray<double> expect0 = {{0, 0}, {0.5, 0.5}, {1, 1}};
        xarray<double> expect1 = {{0, 1}, {0, 1}, {0, 1}};
        ASSERT_TRUE(all(equal(std::get<0>(mesh), expect0)));
        ASSERT_TRUE(all(equal(std::get<1>(mesh), expect1)));
    }

    TEST(xbuilder, meshgrid_arange)
    {
        auto xrange = xt::arange(0, 2);
        auto yrange = xt::arange(0, 2);
        auto grid = xt::meshgrid(xrange, yrange);
        std::ostringstream stream;
        stream << std::get<0>(grid) << std::endl;
    }

    TEST(xbuilder, triu)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::triu(e);

        xarray<double> expected = {{1, 2, 3},
                                   {0, 5, 6},
                                   {0, 0, 9}};

        xarray<double> expected_2 = {{1, 2, 3},
                                     {4, 5, 6},
                                     {0, 8, 9}};

        xarray<double> expected_3 = {{0, 2, 3},
                                     {0, 0, 6},
                                     {0, 0, 0}};

        ASSERT_EQ(size_t(2), t.dimension());
        shape_t expected_shape = {3, 3};
        ASSERT_EQ(expected_shape, t.shape());

        ASSERT_EQ(expected, t);

        xarray<double> t3 = xt::triu(e, 1);
        ASSERT_EQ(expected_3, t3);

        xarray<double> t2 = xt::triu(e, -1);
        ASSERT_EQ(expected_2, t2);
    }

    TEST(xbuilder, tril)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::tril(e);

        xarray<double> expected = {{1, 0, 0},
                                   {4, 5, 0},
                                   {7, 8, 9}};

        xarray<double> expected_2 = {{1, 2, 0},
                                     {4, 5, 6},
                                     {7, 8, 9}};

        xarray<double> expected_3 = {{0, 0, 0},
                                     {4, 0, 0},
                                     {7, 8, 0}};

        ASSERT_EQ(size_t(2), t.dimension());
        shape_t expected_shape = {3, 3};
        ASSERT_EQ(expected_shape, t.shape());

        ASSERT_EQ(expected, t);

        xarray<double> t2 = xt::tril(e, 1);
        ASSERT_EQ(expected_2, t2);

        xarray<double> t3 = xt::tril(e, -1);
        ASSERT_EQ(expected_3, t3);
    }

    TEST(xbuilder, diagonal)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        xarray<double> t = xt::diagonal(e);

        xarray<double> expected = {1, 5, 9};
        ASSERT_EQ(expected, t);

        xt::xarray<double> f = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};

        xarray<double> exp_1 = {1, 5};
        ASSERT_TRUE(all(equal(exp_1, xt::diagonal(f, 1))));
        xarray<double> exp_2 = {0, 4, 8};
        EXPECT_EQ(exp_2, xt::diagonal(f));
        xarray<double> exp_3 = {3, 7, 11};
        EXPECT_EQ(exp_3, xt::diagonal(f, -1));
        xarray<double> exp_4 = {6, 10};
        EXPECT_EQ(exp_4, xt::diagonal(f, -2));
    }

    TEST(xbuilder, diagonal_advanced)
    {
        xarray<double> e = {{{{0, 1, 2}, {3, 4, 5}},
                             {{6, 7, 8}, {9, 10, 11}}},
                            {{{12, 13, 14}, {15, 16, 17}},
                             {{18, 19, 20}, {21, 22, 23}}}};

        xarray<double> d1 = xt::diagonal(e);

        xarray<double> expected = {{{0, 18},
                                    {1, 19},
                                    {2, 20}},
                                   {{3, 21},
                                    {4, 22},
                                    {5, 23}}};
        ASSERT_EQ(expected, d1);

        std::vector<double> d2 = {6, 7, 8, 9, 10, 11};
        xarray<double> expected_2;
        expected_2.resize({2, 3, 1});
        std::copy(d2.begin(), d2.end(), expected_2.template begin<layout_type::row_major>());

        xarray<double> t2 = xt::diagonal(e, 1);
        ASSERT_EQ(expected_2, t2);

        std::vector<double> d3 = {3, 9, 15, 21};
        xarray<double> expected_3;
        expected_3.resize({2, 2, 1});
        std::copy(d3.begin(), d3.end(), expected_3.template begin<layout_type::row_major>());
        xarray<double> t3 = xt::diagonal(e, -1, 2, 3);
        ASSERT_EQ(expected_3, t3);
    }

    TEST(xbuilder, diag)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::diag(xt::diagonal(e));
        xarray<double> expected = xt::eye(3) * e;

        ASSERT_EQ(expected, t);
    }

    TEST(xbuilder, flipud)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 0);
        xarray<double> expected = {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(7, t[idx]);
        ASSERT_EQ(2, t(2, 1));
        ASSERT_EQ(7, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 0);
        xarray<double> expected_2 = {{{6, 7, 8},
                                      {9, 10, 11}},
                                     {{0, 1, 2},
                                      {3, 4, 5}}};
        ASSERT_EQ(expected_2, ft);
    }

    TEST(xbuilder, fliplr)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 1);
        xarray<double> expected = {{3, 2, 1}, {6, 5, 4}, {9, 8, 7}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(3, t[idx]);
        ASSERT_EQ(8, t(2, 1));
        ASSERT_EQ(3, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 1);
        xarray<double> expected_2 = {{{3, 4, 5},
                                      {0, 1, 2}},
                                     {{9, 10, 11},
                                      {6, 7, 8}}};

        ASSERT_EQ(expected_2, ft);
        auto flipped_range = xt::flip(xt::stack(xt::xtuple(arange<double>(2), arange<double>(2))), 1);
        xarray<double> expected_range = {{1, 0}, {1, 0}};
        ASSERT_TRUE(all(equal(flipped_range, expected_range)));
    }
}
