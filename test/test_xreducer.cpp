/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "test_common_macros.hpp"
#if (defined(__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#include "xtensor/xmath.hpp"
#pragma GCC diagnostic pop
#else
#include "xtensor/xmath.hpp"
#endif
#include "xtensor/xutils.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xoptional.hpp"
#include "xtl/xoptional.hpp"

#include "xtensor/xio.hpp"

namespace xt
{

#define CHECK_RESULT_TYPE(EXPRESSION, EXPECTED_TYPE)                             \
{                                                                                \
    using result_type = typename std::decay_t<decltype(EXPRESSION)>::value_type; \
    EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));              \
}

#define CHECK_TAG_TYPE(EXPRESSION, EXPECTED_TYPE)                                    \
{                                                                                    \
    using result_type = typename std::decay_t<decltype(EXPRESSION)>::expression_tag; \
    EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));                  \
}

#define CHECK_TYPE(VALUE, EXPECTED_TYPE)                                         \
{                                                                                \
    using result_type = typename std::decay_t<decltype(VALUE)>;                  \
    EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));              \
}

    template <class T = double>
    struct xreducer_feats
    {
        using axes_type = std::array<std::size_t, 2>;
        using shape_type = typename xarray<T>::shape_type;
        using func = xreducer_functors<std::plus<T>>;

        axes_type m_axes;

        xarray<T> m_a;
    
        xreducer<func, const xarray<T>&, axes_type, xt::reducer_options<T, std::tuple<xt::evaluation_strategy::lazy_type>>> m_red;

        xreducer_feats()
            : m_axes({1, 3}), m_a(ones<T>({3, 2, 4, 6, 5})),
            m_red(func(), m_a, m_axes, xt::evaluation_strategy::lazy)
        {
            for (std::size_t i = 0; i < 2; ++i)
            {
                for (std::size_t j = 0; j < 6; ++j)
                {
                    m_a(1, i, 1, j, 1) = 2;
                }
            }
        }
    };

    template <class T = double>
    struct xreducer_opt_feats
    {
        using axes_type = std::array<std::size_t, 2>;
        using shape_type = typename xarray<xtl::xoptional<T>>::shape_type;
        using func = xreducer_functors<std::plus<xtl::xoptional<T>>>;

        axes_type m_axes;

        xarray<xtl::xoptional<T>> m_a;
    
        xreducer<func, const xarray<xtl::xoptional<T>>&, axes_type, xt::reducer_options<xtl::xoptional<T>, std::tuple<xt::evaluation_strategy::lazy_type>>> m_red;

        xreducer_opt_feats()
            : m_axes({1, 3}), m_a(ones<xtl::xoptional<T>>({3, 2, 4, 6, 5})),
            m_red(func(), m_a, m_axes, xt::evaluation_strategy::lazy)
        {
            for (std::size_t i = 0; i < 2; ++i)
            {
                for (std::size_t j = 0; j < 6; ++j)
                {
                    m_a(1, i, 1, j, 1) = 2;
                }
            }
            m_a(0, 0, 0, 0, 0) = xtl::missing<T>();
        }
    };

    using xreducer_features = xreducer_feats<>;
    using xreducer_opt_features = xreducer_opt_feats<>;


    TEST(xreducer, const_value)
    {
        const_value<int> c_value(10);

        CHECK_RESULT_TYPE(c_value.template rebind<int>(), int);
        CHECK_RESULT_TYPE(c_value.template rebind<float>(), float);
        CHECK_RESULT_TYPE(c_value.template rebind<double>(), double);

        using init_type = xtl::xoptional<int, bool>; 
        const_value<init_type> c_opt_value(xtl::xoptional<int, bool>(20, true));

        CHECK_RESULT_TYPE(c_opt_value, init_type);
        CHECK_RESULT_TYPE(c_opt_value.template rebind<int>(), int);
        CHECK_RESULT_TYPE(c_opt_value.template rebind<float>(), float);
        CHECK_RESULT_TYPE(c_opt_value.template rebind<double>(), double);
    }

    TEST(xreducer, expression_tag)
    {
        // xarray of T
        xarray<int> a1 = {{1, 2, 3}, {4, 5, 6}};
        auto sum1 = xt::sum(a1, {1});
        CHECK_RESULT_TYPE(a1, int);
        CHECK_RESULT_TYPE(sum1, long long);
        CHECK_TAG_TYPE(sum1, xtensor_expression_tag);
        
        // xarray of xoptional<T>
        xarray<xtl::xoptional<int>> a2 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum2 = xt::sum(a2, {1});
        CHECK_RESULT_TYPE(a2, xtl::xoptional<int>);
        CHECK_RESULT_TYPE(sum2, xtl::xoptional<int>);
        CHECK_TAG_TYPE(sum2, xoptional_expression_tag);  // FAIL

        // xarray_optional of T
        xarray_optional<int> a3 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum3 = xt::sum(a3, {1});
        CHECK_RESULT_TYPE(a3, xtl::xoptional<int>);
        CHECK_RESULT_TYPE(sum3, xtl::xoptional<int>);
        CHECK_TAG_TYPE(sum3, xoptional_expression_tag);
    }

    TEST(xreducer, optional)
    {
        // xarray<xtl::xoptional<T>>
        xarray<xtl::xoptional<int>> a1 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum1 = xt::sum(a1, {1});

        CHECK_RESULT_TYPE(sum1, xtl::xoptional<int>);
        CHECK_TYPE(xt::value(sum1)(1), int);  // is currently xtl::xoptional<int>
        EXPECT_EQ(true, xt::has_value(sum1)(0));
        EXPECT_EQ(false, xt::has_value(sum1)(1));  // Fail

        // xarray_optional<T>
        xarray_optional<int> a2 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum2 = xt::sum(a2, {1});

        CHECK_RESULT_TYPE(sum2, xtl::xoptional<int>);
        CHECK_TYPE(xt::value(sum2)(1), int);  // is currently xtl::xoptional<int>
        EXPECT_EQ(true, xt::has_value(sum2)(0));
        EXPECT_EQ(false, xt::has_value(sum2)(1));
    }

    TEST(xreducer, assignment)
    {
        // Nothing computed
        xarray<xtl::xoptional<int>> a1 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum11 = xt::sum(a1, {1});  // OK
        CHECK_RESULT_TYPE(sum11, xtl::xoptional<int>);

        // Computed and assigned in xarray<xoptional<T>>
        xarray<xtl::xoptional<int>> sum12 = xt::sum(a1, {1});  // OK
        CHECK_RESULT_TYPE(sum12, xtl::xoptional<int>);

        // Computed and assigned in xarray_optional<T>
        // xarray_optional<int> sum13 = xt::sum(a1, {1});  // error: invalid static_cast from type 'xtl::xoptional<int, bool>' to type 'int'
        // CHECK_RESULT_TYPE(sum13, xtl::xoptional<int>);

        // Nothing computed
        xarray_optional<int> a2 = {{1, 2, 3}, {4, xtl::missing<int>(), 6}};
        auto sum21 = xt::sum(a2, {1});  // OK
        CHECK_RESULT_TYPE(sum21, xtl::xoptional<int>);

        // Computed and assigned in xarray_optional<T>
        // xarray_optional<int> sum22 = xt::sum(a2, {1});  // error: invalid static_cast from type 'xtl::xoptional<int, bool>' to type 'int'
        // CHECK_RESULT_TYPE(sum22, xtl::xoptional<int>);

        // Computed and assigned in xarray<xoptional<T>>
        // xarray<xtl::xoptional<int>> sum23 = xt::sum(a2, {1});  // error: assignment of read-only location
        // CHECK_RESULT_TYPE(sum23, xtl::xoptional<int>);
    }
/*
    TEST(xreducer, functor_type)
    {
        auto sum = [](auto const& left, auto const& right) { return left + right; };
        auto sum_functor = xt::make_xreducer_functor(sum);
        xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
        xt::xarray<int> a_sums = xt::reduce(std::move(sum_functor), a, {1});
        xt::xarray<int> a_sums2 = xt::reduce(sum_functor, a, {1});
        xt::xarray<int> expect = {6, 15};
        EXPECT_EQ(a_sums, expect);
        EXPECT_EQ(a_sums2, expect);

        xt::xarray<int> a_sums3 = xt::reduce(sum, a, {1});
        EXPECT_EQ(a_sums3, expect);

    }

    TEST(xreducer, errors)
    {
        xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
        XT_EXPECT_THROW(xt::sum(a, {1, 0}), std::runtime_error);
        XT_EXPECT_THROW(xt::sum(a, {0, 2}), std::runtime_error);
        XT_EXPECT_THROW(xt::sum(a, {1, 0}, evaluation_strategy::immediate), std::runtime_error);
        XT_EXPECT_THROW(xt::sum(a, {0, 2}, evaluation_strategy::immediate), std::runtime_error);
    }

    TEST(xreducer, shape)
    {
        xreducer_features features;
        xreducer_features::shape_type s = {3, 4, 5};
        EXPECT_EQ(s, features.m_red.shape());
        EXPECT_EQ(features.m_red.layout(), layout_type::dynamic);
    }

    TEST(xreducer, access)
    {
        xreducer_features features;
        EXPECT_EQ(12, features.m_red(0, 0, 0));
        EXPECT_EQ(24, features.m_red(1, 1, 1));
        EXPECT_EQ(features.m_red(0, 1), features.m_red(0, 0, 1));
        EXPECT_EQ(features.m_red(1, 2, 1, 0, 1), features.m_red(1, 0, 1));

        EXPECT_EQ(12, features.m_red(2, 0, 0, 0));
        EXPECT_EQ(12, features.m_red());
    }

    TEST(xreducer, unchecked)
    {
        xreducer_features features;
        EXPECT_EQ(12, features.m_red.unchecked(0, 0, 0));
        EXPECT_EQ(24, features.m_red.unchecked(1, 1, 1));
    }

    TEST(xreducer, indexed_access)
    {
        xreducer_features features;
        EXPECT_EQ(12, (features.m_red[{0, 0, 0}]));
        EXPECT_EQ(24, (features.m_red[{1, 1, 1}]));
    }

    TEST(xreducer, at)
    {
        xreducer_features features;
        EXPECT_EQ(12, features.m_red.at(0, 0, 0));
        EXPECT_EQ(24, features.m_red.at(1, 1, 1));
        XT_EXPECT_ANY_THROW(features.m_red.at(10, 10, 10));
        XT_EXPECT_ANY_THROW(features.m_red.at(0, 0, 0, 0));
    }

    TEST(xreducer, iterator)
    {
        xreducer_features features;
        auto iter = features.m_red.cbegin();
        auto iter_end = features.m_red.cend();
        const xreducer_features::shape_type& s = features.m_red.shape();
        std::size_t nb_iter = 1;
        nb_iter = std::accumulate(s.cbegin(), s.cend(), nb_iter, std::multiplies<std::size_t>());
        std::advance(iter, static_cast<std::ptrdiff_t>(nb_iter));
        EXPECT_EQ(iter_end, iter);
    }

    TEST(xreducer, assign)
    {
        xreducer_features features;
        xarray<double> res = features.m_red;
        xarray<double> expected = 12 * ones<double>({3, 4, 5});
        expected(1, 1, 1) = 24;
        EXPECT_EQ(expected, res);
    }

    TEST(xreducer, sum)
    {
        xreducer_features features;
        xarray<double> res = xt::sum(features.m_a, features.m_axes);
        xarray<double> expected = 12 * xt::ones<double>({3, 4, 5});
        expected(1, 1, 1) = 24;
        EXPECT_EQ(expected, res);
        CHECK_RESULT_TYPE(res, double);

        xreducer_opt_features opt_features;
        xarray<xtl::xoptional<double>> opt_res = xt::sum(opt_features.m_a, opt_features.m_axes);
        xarray<xtl::xoptional<double>> opt_expected = 12 * xt::ones<double>({3, 4, 5});
        opt_expected(1, 1, 1) = 24;
        opt_expected(0, 0, 0) = xtl::missing<double>();
        EXPECT_EQ(opt_expected, opt_res);
        CHECK_RESULT_TYPE(opt_res, xtl::xoptional<double>);
    }

    TEST(xreducer, sum_tensor)
    {
        xtensor<double, 2> m = {{1, 2}, {3, 4}};
        xarray<double> res = xt::sum(m, {0});
        EXPECT_EQ(res.dimension(), std::size_t(1));
        EXPECT_EQ(res(0), 4.0);
        EXPECT_EQ(res(1), 6.0);
        CHECK_RESULT_TYPE(res, double);

        xtensor<xtl::xoptional<double>, 2> opt_m = {{1, 2}, {3, 4}};
        xarray<xtl::xoptional<double>> opt_res = xt::sum(m, {0});
        EXPECT_EQ(opt_res.dimension(), std::size_t(1));
        EXPECT_EQ(opt_res(0), 4.0);
        EXPECT_EQ(opt_res(1), 6.0);
        CHECK_RESULT_TYPE(opt_res, xtl::xoptional<double>);
    }

    TEST(xreducer, single_axis_sugar)
    {
        xarray<double> m = {{1, 0}, {3, 4}};
/*
        xarray<std::size_t> res1 = xt::count_nonzero(m, {1});
        xarray<std::size_t> res2 = xt::count_nonzero(m, 1);
        EXPECT_EQ(res1, res2);
        CHECK_RESULT_TYPE(res1, std::size_t);
        CHECK_RESULT_TYPE(res2, std::size_t);

        xarray<double> res3 = xt::sum(m, {1});
        xarray<double> res4 = xt::sum(m, 1);
        EXPECT_EQ(res3, res4);
        CHECK_RESULT_TYPE(res3, double);
        CHECK_RESULT_TYPE(res4, double);

        xt::xarray_optional<double> m_opt = {{1, 0}, {3, xtl::missing<double>()}};

        xarray<xtl::xoptional<std::size_t>> opt_res1 = xt::count_nonzero(m_opt, {1});/*
        xarray<std::size_t> opt_res2 = xt::count_nonzero(m_opt, 1);
        EXPECT_EQ(opt_res1, opt_res2);
        CHECK_RESULT_TYPE(opt_res1, double);
        CHECK_RESULT_TYPE(opt_res2, double);

        xarray<xtl::xoptional<double>> opt_res3 = xt::sum(m_opt, {1});
        xarray<xtl::xoptional<double>> opt_res4 = xt::sum(m_opt, 1);
        EXPECT_EQ(opt_res3, opt_res4);
        CHECK_RESULT_TYPE(opt_res3, double);
        CHECK_RESULT_TYPE(opt_res4, double);


    }
    */
/*
    TEST(xreducer, sum2)
    {
        xarray<double> u = ones<double>({2, 4});
        xarray<double> expectedu0 = 2 * ones<double>({4});
        xarray<double> resu0 = sum(u, {0});
        EXPECT_EQ(expectedu0, resu0);
        xarray<double> expectedu1 = 4 * ones<double>({2});
        xarray<double> resu1 = sum(u, {std::size_t(1)});
        xarray<double> resm1 = sum(u, {-1});

        std::array<std::size_t, 1> a_us = {1};
        std::array<std::ptrdiff_t, 1> a_ss = {-1};
        xarray<double> res_refu1 = sum(u, a_us);
        xarray<double> res_refm1 = sum(u, a_ss);

        EXPECT_EQ(expectedu1, resm1);
        EXPECT_EQ(expectedu1, resu1);
        EXPECT_EQ(expectedu1, res_refm1);
        EXPECT_EQ(expectedu1, res_refu1);

        xarray<double> v = ones<double>({4, 2});
        xarray<double> expectedv0 = 4 * ones<double>({2});
        xarray<double> resv0 = sum(v, {0});
        EXPECT_EQ(expectedv0, resv0);
        xarray<double> expectedv1 = 2 * ones<double>({4});
        xarray<double> resv1 = sum(v, {1});
        EXPECT_EQ(expectedv1, resv1);

        // check that there is no overflow
        xarray<uint8_t> c = ones<uint8_t>({1000});
        EXPECT_EQ(1000u, sum(c)());
    }
*/
/*
    TEST(xreducer, sum_all)
    {
        xreducer_features features;
        auto res = xt::sum(features.m_a);
        double expected = 732;
        CHECK_RESULT_TYPE(res, double);
        EXPECT_EQ(res(), expected);

        xreducer_opt_features opt_features;
        auto opt_res = xt::sum(opt_features.m_a);
        auto opt_expected = xtl::missing<double>();
        CHECK_RESULT_TYPE(opt_res, xtl::xoptional<double>);
        EXPECT_EQ(opt_res(), opt_expected);
    }

    TEST(xreducer, prod)
    {
        // check that there is no overflow
        xarray<uint8_t> c = 2 * xt::ones<uint8_t>({34});
        auto res = xt::prod(c);
        CHECK_RESULT_TYPE(res, uint8_t);  // TODO: fail, investigate
        EXPECT_EQ(1ULL << 34, res());

        xarray<double> d = 2 * xt::ones<double>({7});
        auto res2 = xt::prod(d);
        CHECK_RESULT_TYPE(res2, double);
        EXPECT_EQ(128, res2());

        xreducer_opt_features opt_features;
        auto opt_res = xt::prod(opt_features.m_a);
        auto opt_expected = xtl::missing<double>();
        CHECK_RESULT_TYPE(opt_res, xtl::xoptional<double>);
        EXPECT_EQ(opt_res(), opt_expected);

    }

    TEST(xreducer, mean)
    {
        xtensor<double, 2> input
            {{-1.0, 0.0}, {1.0, 0.0}};
        auto mean_all = mean(input);
        auto mean0 = mean(input, {0});
        auto mean1 = mean(input, {1});

        xtensor<double, 0> expect_all = 0.0;
        xtensor<double, 1> expect0 = {0.0, 0.0};
        xtensor<double, 1> expect1 = {-0.5, 0.5};

        EXPECT_EQ(mean_all(), expect_all());
        EXPECT_TRUE(all(equal(mean0, expect0)));
        EXPECT_TRUE(all(equal(mean1, expect1)));

        xarray<uint8_t> c = {1, 2};
        EXPECT_EQ(mean(c)(), 1.5);

        const auto rvalue_xarray = [] () { return xtensor<double, 1>({1, 2}); };
        EXPECT_EQ(mean(rvalue_xarray(), {0})(), 1.5);
    }

    TEST(xreducer, average)
    {
        xt::xtensor<float, 2> a = {{ 3, 4, 2, 1}, { 1, 1, 3, 2}};
        xt::xarray<double> all_weights = {{1, 2, 3, 4}, { 5, 6, 7, 8}};
        auto avg_all = xt::average(a, all_weights);
        auto avg_all_2 = xt::average(a, all_weights, {0ul, 1ul});

        auto avg0 = xt::average(a, xt::xarray<double>{3, 9}, {0});
        auto avg1 = xt::average(a, xt::xarray<double>{1,2,3,4}, {1});
        auto avg_m1 = xt::average(a, xt::xarray<double>{1,2,3,4}, {-1});
        auto avg_d1 = xt::average(a, xt::xarray<double>{1,2,3,4}, {-1}, evaluation_strategy::immediate);

        xtensor<double, 0> expect_all = 1.9166666666666667;
        xtensor<double, 1> expect0 = {1.5, 1.75, 2.75, 1.75};
        xtensor<double, 1> expect1 = {2.1, 2.0};

        EXPECT_TRUE(allclose(avg_all, expect_all));
        EXPECT_TRUE(allclose(avg_all_2, expect_all));
        EXPECT_TRUE(all(equal(avg0, expect0)));
        EXPECT_TRUE(all(equal(avg1, expect1)));
        EXPECT_TRUE(all(equal(avg_m1, expect1)));
        EXPECT_TRUE(all(equal(avg_d1, expect1)));
    }

    TEST(xreducer, minmax)
    {
        using A = std::array<double, 2>;

        xtensor<double, 2> input
            {{-1.0, 0.0}, {1.0, 0.0}};
        EXPECT_EQ(minmax(input)(), (A{-1.0, 1.0}));
    }

    TEST(xreducer, immediate)
    {
        xarray<double> a = xt::arange(27);
        a.resize({3, 3, 3});

        xarray<double> a_lz = sum(a);
        auto a_gd = sum(a, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1});
        a_gd = sum(a, {1}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {0, 2});
        a_gd = sum(a, {0, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1, 2});
        a_gd = sum(a, {1, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a = xt::arange(4 * 3 * 6 * 2 * 7);
        a.resize({4, 3, 6, 2, 7});

        a_lz = sum(a);
        a_gd = sum(a, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1});
        a_gd = sum(a, {1}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {0, 2});
        a_gd = sum(a, {0, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1, 2});
        a_gd = sum(a, {1, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1, 3, 4});
        a_gd = sum(a, {1, 3, 4}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {0, 1, 4});
        a_gd = sum(a, {0, 1, 4}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {0, 1, 3});
        a_gd = sum(a, {0, 1, 3}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {0, 2, 3});
        a_gd = sum(a, {0, 2, 3}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        a_lz = sum(a, {1, 2, 3});
        a_gd = sum(a, {1, 2, 3}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        xtensor<short, 3, layout_type::column_major> ct = xt::random::randint<short>({1, 5, 3});
        EXPECT_EQ(sum(ct, {0, 2}), sum(ct, {0, 2}, evaluation_strategy::immediate));

        xtensor<short, 5, layout_type::column_major> ct2 = xt::random::randint<short>({1, 5, 1, 2, 3});
        EXPECT_EQ(sum(ct2, {0, 1, 2}), sum(ct2, {0, 1, 2}, evaluation_strategy::immediate));
        EXPECT_EQ(sum(ct2, {2, 3}), sum(ct2, {2, 3}, evaluation_strategy::immediate));
        EXPECT_EQ(sum(ct2, {1, 3}), sum(ct2, {1, 3}, evaluation_strategy::immediate));
    }

    TEST(xreducer, chaining_reducers)
    {
        xt::xarray<double> a = {{ 1., 2. },
                                { 3., 4. }};

        auto b = a - xt::sum(a, { 0 });
        auto c = xt::sum(b, { 0 });
        EXPECT_EQ(c(0), -4.);
        EXPECT_EQ(c(1), -6.);
    }

    TEST(xreducer, immediate_shape)
    {
        xtensor<double, 2> c = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        auto xa = xt::sum(c, {0}, evaluation_strategy::immediate);
        auto is_arr = [](const auto& c)
        {
            bool istrue = detail::is_array<std::decay_t<decltype(c)>>::value;
            return istrue;
        };

    #ifndef X_OLD_CLANG
        EXPECT_TRUE(is_arr(xa.shape()));
    #endif

        xtensor<double, 3> a;
        a.resize({3, 3, 3});
        std::iota(a.storage().begin(), a.storage().end(), 0);

        xarray<double> a_lz = sum(a);
        auto a_gd = sum(a, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);
        EXPECT_TRUE(is_arr(a_gd.shape()));

        a_lz = sum(a, {1});
        auto a_gd_1 = sum(a, {1}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_1);

        a_lz = sum(a, {0, 2});
        auto a_gd_2 = sum(a, {0, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_2);

    #ifndef X_OLD_CLANG
        EXPECT_TRUE(is_arr(a_gd_1.shape()));
        EXPECT_TRUE(is_arr(a_gd_2.shape()));
    #endif

        a_lz = sum(a, {1, 2});
        a_gd_2 = sum(a, {1, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_2);
    }

    TEST(xreducer, xfixed_reduction)
    {
        xtensor_fixed<double, xshape<3, 3, 3>> a;
        std::iota(a.storage().begin(), a.storage().end(), 0);

        xtensor<double, 3> b;
        b.resize({3, 3, 3});
        std::iota(b.storage().begin(), b.storage().end(), 0);

        auto is_arr = [](const auto& c)
        {
            bool istrue = detail::is_array<std::decay_t<decltype(c)>>::value;
            return istrue;
        };

        xarray<double> a_lz = sum(a);
        auto a_gd = sum(a, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd);

        // EXPECT_TRUE(is_fixed(a_gd.shape())); // this actually evaluates to const_array

        a_lz = sum(a, xt::dynamic_shape<std::size_t>{1});
        auto a_gd_1 = sum(a, xt::dynamic_shape<std::size_t>{1}, evaluation_strategy::immediate);
        auto b_gd_1 = sum(b, {1}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_1);
        EXPECT_EQ(a_lz, b_gd_1);

        a_lz = sum(a, {0, 2});
        auto a_gd_2 = sum(a, {0, 2}, evaluation_strategy::immediate);
        auto b_gd_2 = sum(b, {0, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_2);
        EXPECT_EQ(b_gd_2, a_gd_2);
        EXPECT_EQ(a_gd_2.dimension(), std::size_t(1));

    #ifndef X_OLD_CLANG
        // EXPECT_TRUE(is_arr(a_gd_1.shape()));
        EXPECT_TRUE(is_arr(a_gd_2.shape()));
    #endif

        a_lz = sum(a, {1, 2});
        a_gd_2 = sum(a, {1, 2}, evaluation_strategy::immediate);
        EXPECT_EQ(a_lz, a_gd_2);

        auto a_lx_3 = sum(a, {1, 2});
        auto a_lz_3 = sum(a, xshape<1, 2>());
        auto a_gd_3 = sum(a, xshape<1, 2>(), evaluation_strategy::immediate);
        xarray<double> xevd = a_lz_3;
        EXPECT_EQ(a_lz_3, a_gd_2);
        EXPECT_TRUE(a_gd_3 == a_gd_2);
        bool truth = std::is_same<decltype(a_gd_3), xtensor_fixed<double, xshape<3>>>::value;
        EXPECT_TRUE(truth);

        xtensor<short, 3> ct = xt::random::randint<short>({1, 5, 3});
        xtensor_fixed<short, xshape<1, 5, 3>> c = ct;
        auto b_fx_1 = sum(c, xshape<0, 2>(), evaluation_strategy::immediate);
        auto b_fx_2 = sum(c, xshape<0, 1>(), evaluation_strategy::immediate);
        auto b_fx_3 = sum(c, xshape<0, 1, 2>(), evaluation_strategy::immediate);

        EXPECT_EQ(sum(ct, {0, 2}, evaluation_strategy::immediate), sum(c, {0, 2}));
        EXPECT_TRUE(b_fx_1 == sum(c, {0, 2}));
        EXPECT_TRUE(b_fx_2 == sum(c, {0, 1}));
        EXPECT_EQ(b_fx_3, sum(c, {0, 1, 2}));

        truth = std::is_same<std::decay_t<decltype(b_fx_1)>, xtensor_fixed<long long, xshape<5>>>::value;
        EXPECT_TRUE(truth);
        truth = std::is_same<std::decay_t<decltype(b_fx_3)>, xtensor_fixed<long long, xshape<>>>::value;
        EXPECT_TRUE(truth);

        truth = std::is_same<xshape<1, 3>, typename fixed_xreducer_shape_type<xshape<1, 5, 3>, xshape<1>>::type>();
        EXPECT_TRUE(truth);
        truth = std::is_same<xshape<5>, typename fixed_xreducer_shape_type<xshape<1, 5, 3>, xshape<0, 2>>::type>();
        EXPECT_TRUE(truth);
        truth = std::is_same<xshape<>, typename fixed_xreducer_shape_type<xshape<1, 5, 3>, xshape<0, 1, 2>>::type>();
        EXPECT_TRUE(truth);
        truth = std::is_same<xshape<1, 5>, typename fixed_xreducer_shape_type<xshape<1, 5>, xshape<2>>::type>();
        EXPECT_TRUE(truth);
    }

    TEST(xreducer, view_steppers)
    {
        xt::xtensor<double, 2> X({10, 20});
        xt::xtensor<double, 2> Y(X.shape());

        X = xt::random::randn<double>(X.shape());

        xt::xtensor<double, 2> vx0 = xt::view(xt::sum(X, {1}), xt::all(), xt::newaxis());
        xt::xtensor<double, 2> vx1 = xt::expand_dims(xt::sum(X, {1}), 1);

        EXPECT_EQ(vx0, vx1);
    }

    TEST(xreducer, wrong_number_of_indices)
    {
        xt::xtensor<double, 4> a = xt::random::rand<double>({5, 5, 5, 5});
        double e = xt::sum(a)();
        double s1 = xt::sum(a)(0);
        EXPECT_EQ(s1, e);
        double s2 = xt::sum(a)(0, 1, 2, 3, 4, 5, 0);
        EXPECT_EQ(s2, e);

        auto red = xt::sum(a, {0});
        EXPECT_EQ(red(2), red(0, 0, 2));
        EXPECT_EQ(red(1, 2), red(0, 1, 2));
        EXPECT_EQ(red(1, 2), red(1, 1, 1, 1, 1, 0, 1, 2));
    }

    TEST(xreducer, normalize_axes)
    {
        xt::xtensor<double, 4> x{};
        std::vector<std::size_t> sva {0, 1, 2, 3};
        std::vector<std::ptrdiff_t> svb = {-4, 1, -2, -1};
        std::initializer_list<std::ptrdiff_t> initlist = {-4, 1, -2, -1};
        std::array<std::size_t, 4> saa = {0, 1, 2, 3};
        std::array<std::ptrdiff_t, 4> sab = {-4, 1, -2, -1};

        auto resa = forward_normalize<std::vector<std::size_t>>(x, svb);
        EXPECT_EQ(forward_normalize<std::vector<std::size_t>>(x, svb), sva);
        EXPECT_EQ(forward_normalize<std::vector<std::size_t>>(x, initlist), sva);
        auto resaa = forward_normalize<std::array<std::size_t, 4>>(x, saa);
        EXPECT_EQ(resaa, saa);
        auto resab = forward_normalize<std::array<std::size_t, 4>>(x, sab);
        EXPECT_EQ(resab, saa);
    }

    TEST(xreducer, input_0d)
    {
        xt::xarray<double> a;
        EXPECT_EQ(0., xt::amin(a)[0]);

        using A = std::array<double, 2>;
        xt::xarray<double> b(1.2);
        EXPECT_EQ(b.dimension(), 0u);
        EXPECT_EQ(minmax(b)(), (A{1.2, 1.2}));
    }
    
    template <std::size_t... I, std::size_t... J>
    bool operator==(fixed_shape<I...>, fixed_shape<J...>)
    {
        std::array<std::size_t, sizeof...(I)> ix = {I...};
        std::array<std::size_t, sizeof...(J)> jx = {J...};
        return sizeof...(J) == sizeof...(I) && std::equal(ix.begin(), ix.end(), jx.begin());
    }

    TEST(xreducer, keep_dims)
    {
        xt::xtensor<double, 4> a = xt::reshape_view(xt::arange<double>(5 * 5 * 5 * 5), {5, 5, 5, 5});

    #ifndef X_OLD_CLANG
        auto res = xt::sum(a, {0, 1}, xt::keep_dims | xt::evaluation_strategy::immediate);   
        EXPECT_EQ(res.shape(), (std::array<std::size_t, 4>{1, 1, 5, 5}));
        auto res2 = xt::sum(a, {0, 1}, xt::keep_dims);   
        EXPECT_EQ(res2.shape(), (std::array<std::size_t, 4>{1, 1, 5, 5}));
    #else
        auto res = xt::sum(a, {0, 1}, xt::keep_dims | xt::evaluation_strategy::immediate);   
        EXPECT_EQ(res.shape(), (xt::dynamic_shape<std::size_t>{1, 1, 5, 5}));
        auto res2 = xt::sum(a, {0, 1}, xt::keep_dims);   
        EXPECT_EQ(res2.shape(), (xt::dynamic_shape<std::size_t>{1, 1, 5, 5}));
    #endif

        xt::xarray<double> b = a;
        auto res3 = xt::sum(b, {0, 1}, xt::keep_dims | xt::evaluation_strategy::immediate);   
        EXPECT_EQ(res3.shape(), (xt::dynamic_shape<std::size_t>{1, 1, 5, 5}));
        auto res4 = xt::sum(b, {0, 1}, xt::keep_dims | xt::evaluation_strategy::lazy);   
        EXPECT_EQ(res4.shape(), (xt::dynamic_shape<std::size_t>{1, 1, 5, 5}));

        xt::xarray<double> resx3 = xt::sum(a, {0, 1});   
        auto exp1 = xt::sum(a, {0, 1});

        EXPECT_EQ(res, res2);
        EXPECT_EQ(res, res3);
        EXPECT_EQ(res, res4);

        EXPECT_TRUE(xt::allclose(res, res2));
        EXPECT_TRUE(xt::allclose(res, res3));
        EXPECT_TRUE(xt::allclose(res, res4));

        auto res5 = xt::sum(a, xt::keep_dims);
        EXPECT_EQ(res5.shape(), (xt::static_shape<size_t, 4>{1, 1, 1, 1}));
        auto res6 = xt::sum(a);
        EXPECT_EQ(res6.shape(), (xt::static_shape<size_t, 0>{}));
        xt::xtensor_fixed<double, xshape<5, 5, 5, 5>> c = a;
        auto res7 = xt::sum(c);
        EXPECT_EQ(res7.shape(), (xt::xshape<>{}));
        auto res8 = xt::sum(c, xt::keep_dims);
        EXPECT_EQ(res8.shape(), (xt::xshape<1, 1, 1, 1>{}));
    }

    TEST(xreducer, initial_value)
    {
        xt::xtensor<double, 4> a = xt::reshape_view(xt::arange<double>(5 * 5 * 5 * 5), {5, 5, 5, 5});

        auto res = xt::sum(a, {0, 2}, xt::keep_dims | xt::evaluation_strategy::immediate | initial(5));   
        auto reso = xt::sum(a, {0, 2}, xt::keep_dims | xt::evaluation_strategy::immediate);
        EXPECT_EQ(res, reso + 5);

        xt::xarray<double> res2 = xt::sum(a, {0, 2}, xt::keep_dims | initial(5));   
        auto reso2 = xt::sum(a, {0, 2}, xt::keep_dims);   
        EXPECT_EQ(res2, reso2 + 5);

        auto re0 = xt::prod(a, {1, 2}, xt::keep_dims | xt::evaluation_strategy::immediate | initial(0));   
        EXPECT_TRUE(xt::all(equal(re0, 0.)));

        auto rex0 = xt::prod(a, {1, 2}, initial(0));
        EXPECT_TRUE(xt::all(equal(rex0, 0.)));
    }

    TEST(xreducer, ones_first)
    {
        auto a = xt::ones<int>(std::vector<int>({ 1,1,2,4 }));
        std::vector<int> arraxis = { 1 };
        auto result = xt::sum(a, arraxis, xt::keep_dims | xt::evaluation_strategy::immediate);
        EXPECT_EQ(a, result);

        std::vector<int> arraxis2 = { 1, 2 };
        auto res2 = xt::sum(a, arraxis2, xt::keep_dims | xt::evaluation_strategy::immediate);
        xt::xarray<int> expected = { {{{2, 2, 2, 2}}} };
        EXPECT_EQ(expected, res2);
    }

    TEST(xreducer, empty_axes)
    {
        xarray<int> a = { {1, 2, 3}, {4, 5, 6} };
        std::vector<std::size_t> axes = {};
        auto res0 = xt::sum(a, axes);
        auto res1 = xt::sum(a, axes, xt::keep_dims | xt::evaluation_strategy::immediate);

        EXPECT_EQ(res0, a);
        EXPECT_EQ(res1, a);
    }

    TEST(xreducer, zero_shape)
    {
        xt::xarray<int> x = xt::zeros<int>({ 0, 1 });
        
        auto res0 = xt::sum(x, { 0 }, xt::keep_dims);
        EXPECT_EQ(res0.shape()[0], size_t(1));
        EXPECT_EQ(res0.shape()[1], size_t(1));
        EXPECT_EQ(res0(0, 0), 0);

        auto res1 = xt::sum(x, { 1 }, xt::keep_dims);
        EXPECT_EQ(res1.shape()[0], size_t(0));
        EXPECT_EQ(res1.shape()[1], size_t(1));
        EXPECT_EQ(res1.size(), size_t(0));

        auto res2 = xt::sum(x, xt::keep_dims);
        EXPECT_EQ(res2.shape()[0], size_t(1));
        EXPECT_EQ(res2.shape()[1], size_t(1));
        EXPECT_EQ(res2(0, 0), 0);
    }

    TEST(xreducer, empty_array)
    {
        xt::xarray<double> a = xt::ones<double>({ 1, 2, 0, 1 });
        double result0 = xt::mean(a)();
        EXPECT_TRUE(std::isnan(result0));

        auto result1 = xt::mean(a, { 1 });
        auto expected1 = xt::xarray<double>::from_shape({1, 0, 1});
        EXPECT_EQ(result1, expected1);
       
        auto result2 = xt::mean(a, { 2 });
        auto expected2 = xt::xarray<double>::from_shape({1, 2, 1});
        EXPECT_EQ(result2.shape(), expected2.shape());
        EXPECT_TRUE(std::isnan(result2(0, 0, 0)));
        EXPECT_TRUE(std::isnan(result2(0, 1, 0)));
    }

    TEST(xreducer, double_axis)
    {
        xt::xarray<int> a = xt::ones<int>({ 3, 2});
        XT_EXPECT_ANY_THROW(xt::sum(a, {1, 1}));
    }
*/
    
/*
    TEST(xreducer, sum_xtensor_of_fixed)
    {
        xt::xtensor_fixed<float, xt::xshape<3>> a = {1, 2, 3}, b = {1, 2, 3};
        xt::xtensor<xt::xtensor_fixed<float, xt::xshape<3>>, 1> c = {a, b};
        auto res = xt::sum(c)();
        EXPECT_EQ(res, a * 2.);
    }
*/
}

