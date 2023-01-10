/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/


#include <cstddef>

#include "xtensor/xarray.hpp"
#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xtensor.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    template <class C, std::size_t>
    struct redim_container
    {
        using type = C;
    };

    template <class T, std::size_t N, layout_type L, std::size_t NN>
    struct redim_container<xtensor<T, N, L>, NN>
    {
        using type = xtensor<T, NN, L>;
    };

    template <class C, std::size_t N>
    using redim_container_t = typename redim_container<C, N>::type;

    namespace xop_test
    {
        template <class C, class T>
        struct rebind_container;

        template <class T, class NT>
        struct rebind_container<xarray<T>, NT>
        {
            using type = xarray<NT>;
        };

        template <class T, std::size_t N, class NT>
        struct rebind_container<xtensor<T, N>, NT>
        {
            using type = xtensor<NT, N>;
        };

        template <class C, class T>
        using rebind_container_t = typename rebind_container<C, T>::type;
    }

    class my_double
    {
    public:

        my_double(double d = 0.)
            : m_value(d)
        {
        }

        double& operator+=(double rhs)
        {
            m_value += rhs;
            return m_value;
        }

    private:

        double m_value;
    };

    double operator+(double rhs, my_double lhs)
    {
        my_double tmp(lhs);
        return tmp += rhs;
    }

    template <class C>
    class operation : public ::testing::Test
    {
    public:

        using storage_type = C;
    };


    template <class T>
    struct int_rebind;

    template <>
    struct int_rebind<xarray<double>>
    {
        using type = xarray<int>;
    };

    template <>
    struct int_rebind<xtensor<double, 2>>
    {
        using type = xtensor<int, 2>;
    };

    template <class T>
    using int_rebind_t = typename int_rebind<T>::type;

    struct vtype
    {
        double a = 0;
        size_t b = 0;

        explicit operator double() const
        {
            return a;
        }
    };

    using xarray_type = xarray<double>;
    using xtensor_2_type = xtensor<double, 2>;

#define XOPERATION_TEST_TYPES xarray_type, xtensor_2_type

    TEST_SUITE("operation")
    {
        TEST_CASE_TEMPLATE("plus", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            double ref = +(a(0, 0));
            double actual = (+a)(0, 0);
            EXPECT_EQ(ref, actual);
        }

        TEST_CASE_TEMPLATE("minus", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            double ref = -(a(0, 0));
            double actual = (-a)(0, 0);
            EXPECT_EQ(ref, actual);
        }

        TEST_CASE_TEMPLATE("add", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            TypeParam b(shape, 1.3);
            EXPECT_EQ((a + b)(0, 0), a(0, 0) + b(0, 0));

            double sb = 1.2;
            EXPECT_EQ((a + sb)(0, 0), a(0, 0) + sb);

            double sa = 4.6;
            EXPECT_EQ((sa + b)(0, 0), sa + b(0, 0));
        }

        TEST_CASE_TEMPLATE("subtract", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            TypeParam b(shape, 1.3);
            EXPECT_EQ((a - b)(0, 0), a(0, 0) - b(0, 0));

            double sb = 1.2;
            EXPECT_EQ((a - sb)(0, 0), a(0, 0) - sb);

            double sa = 4.6;
            EXPECT_EQ((sa - b)(0, 0), sa - b(0, 0));
        }

        TEST_CASE_TEMPLATE("multiply", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            TypeParam b(shape, 1.3);
            EXPECT_EQ((a * b)(0, 0), a(0, 0) * b(0, 0));

            double sb = 1.2;
            EXPECT_EQ((a * sb)(0, 0), a(0, 0) * sb);

            double sa = 4.6;
            EXPECT_EQ((sa * b)(0, 0), sa * b(0, 0));
        }

        TEST_CASE_TEMPLATE("divide", TypeParam, XOPERATION_TEST_TYPES)
        {
            using shape_type = typename TypeParam::shape_type;
            shape_type shape = {3, 2};
            TypeParam a(shape, 4.5);
            TypeParam b(shape, 1.3);
            EXPECT_EQ((a / b)(0, 0), a(0, 0) / b(0, 0));

            double sb = 1.2;
            EXPECT_EQ((a / sb)(0, 0), a(0, 0) / sb);

            double sa = 4.6;
            EXPECT_EQ((sa / b)(0, 0), sa / b(0, 0));
        }

        TEST_CASE_TEMPLATE("modulus", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container = xop_test::rebind_container_t<TypeParam, int>;
            using shape_type = typename int_container::shape_type;

            shape_type shape = {3, 2};
            int_container a(shape, 11);
            int_container b(shape, 3);
            EXPECT_EQ((a % b)(0, 0), a(0, 0) % b(0, 0));

            int sb = 3;
            EXPECT_EQ((a % sb)(0, 0), a(0, 0) % sb);

            int sa = 11;
            EXPECT_EQ((sa % b)(0, 0), sa % b(0, 0));
        }


        TEST_CASE_TEMPLATE("bitwise_and", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_tensor = int_rebind_t<TypeParam>;
            using shape_type = typename int_tensor::shape_type;
            shape_type shape = {3, 2};
            int_tensor a(shape, 14);
            int_tensor b(shape, 15);
            EXPECT_EQ((a & b)(0, 0), a(0, 0) & b(0, 0));

            int sb = 48;
            EXPECT_EQ((a & sb)(0, 0), a(0, 0) & sb);

            int sa = 24;
            EXPECT_EQ((sa & b)(0, 0), sa & b(0, 0));
        }

        TEST_CASE_TEMPLATE("bitwise_or", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_tensor = int_rebind_t<TypeParam>;
            using shape_type = typename int_tensor::shape_type;
            shape_type shape = {3, 2};
            int_tensor a(shape, 14);
            int_tensor b(shape, 15);
            EXPECT_EQ((a | b)(0, 0), a(0, 0) | b(0, 0));

            int sb = 48;
            EXPECT_EQ((a | sb)(0, 0), a(0, 0) | sb);

            int sa = 24;
            EXPECT_EQ((sa | b)(0, 0), sa | b(0, 0));
        }

        TEST_CASE_TEMPLATE("bitwise_xor", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_tensor = int_rebind_t<TypeParam>;
            using shape_type = typename int_tensor::shape_type;
            shape_type shape = {3, 2};
            int_tensor a(shape, 14);
            int_tensor b(shape, 15);
            EXPECT_EQ((a ^ b)(0, 0), a(0, 0) ^ b(0, 0));

            int sb = 48;
            EXPECT_EQ((a ^ sb)(0, 0), a(0, 0) ^ sb);

            int sa = 24;
            EXPECT_EQ((sa ^ b)(0, 0), sa ^ b(0, 0));
        }

        TEST_CASE_TEMPLATE("bitwise_not", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_tensor = int_rebind_t<TypeParam>;
            using shape_type = typename int_tensor::shape_type;
            shape_type shape = {3, 2};
            int_tensor a(shape, 15);
            EXPECT_EQ((~a)(0, 0), ~(a(0, 0)));
        }

        TEST_CASE_TEMPLATE("less", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {1, 1, 1, 0, 0};
            bool_container b = a < 4;
            EXPECT_EQ(expected, b);
            bool_container b2 = less(a, 4);
            EXPECT_EQ(expected, b2);
        }

        TEST_CASE_TEMPLATE("less_equal", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {1, 1, 1, 1, 0};
            bool_container b = a <= 4;
            EXPECT_EQ(expected, b);
            bool_container b2 = less_equal(a, 4);
            EXPECT_EQ(expected, b2);
        }

        TEST_CASE_TEMPLATE("greater", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {0, 0, 0, 0, 1};
            bool_container b = a > 4;
            EXPECT_EQ(expected, b);
            bool_container b2 = greater(a, 4);
            EXPECT_EQ(expected, b2);
        }

        TEST_CASE_TEMPLATE("greater_equal", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {0, 0, 0, 1, 1};
            bool_container b = a >= 4;
            EXPECT_EQ(expected, b);
            bool_container b2 = greater_equal(a, 4);
            EXPECT_EQ(expected, b2);
        }

        TEST_CASE_TEMPLATE("negate", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {1, 1, 1, 0, 0};
            bool_container b = !(a >= 4);
            EXPECT_EQ(expected, b);
        }

        TEST_CASE_TEMPLATE("equal", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {0, 0, 0, 1, 0};
            bool_container b = equal(a, 4);
            EXPECT_EQ(expected, b);

            container_1d other = {1, 2, 3, 0, 0};
            bool_container b_2 = equal(a, other);
            bool_container expected_2 = {1, 1, 1, 0, 0};
            EXPECT_EQ(expected_2, b_2);
        }

        TEST_CASE_TEMPLATE("not_equal", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            container_1d a = {1, 2, 3, 4, 5};
            bool_container expected = {1, 1, 1, 0, 1};
            bool_container b = not_equal(a, 4);
            EXPECT_EQ(expected, b);

            container_1d other = {1, 2, 3, 0, 0};
            bool_container b_2 = not_equal(a, other);
            bool_container expected_2 = {0, 0, 0, 1, 1};
            EXPECT_EQ(expected_2, b_2);
        }

        TEST_CASE_TEMPLATE("logical_and", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            bool_container a = {0, 0, 0, 1, 0};
            bool_container expected = {0, 0, 0, 0, 0};
            bool_container b = a && false;
            bool_container c = a && a;
            EXPECT_EQ(expected, b);
            EXPECT_EQ(c, a);
        }

        TEST_CASE_TEMPLATE("logical_or", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using bool_container = xop_test::rebind_container_t<container_1d, bool>;
            bool_container a = {0, 0, 0, 1, 0};
            bool_container other = {0, 0, 0, 0, 0};
            bool_container b = a || other;
            bool_container c = a || false;
            bool_container d = a || true;
            EXPECT_EQ(b, a);
            EXPECT_EQ(c, a);

            bool_container expected = {1, 1, 1, 1, 1};
            EXPECT_EQ(expected, d);
        }

        TEST_CASE_TEMPLATE("any", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container = xop_test::rebind_container_t<container_1d, int>;
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            int_container a = {0, 0, 3};
            EXPECT_EQ(true, any(a));
            int_container_2d b = {{0, 0, 0}, {0, 0, 0}};
            EXPECT_EQ(false, any(b));
        }

        TEST_CASE_TEMPLATE("minimum", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container = xop_test::rebind_container_t<container_1d, int>;
            int_container a = {0, 0, 3};
            int_container b = {-1, 0, 10};
            int_container expected = {-1, 0, 3};
            EXPECT_TRUE(all(equal(minimum(a, b), expected)));
        }

        TEST_CASE_TEMPLATE("maximum", TypeParam, XOPERATION_TEST_TYPES)
        {
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container = xop_test::rebind_container_t<container_1d, int>;
            int_container a = {0, 0, 3};
            int_container b = {-1, 0, 10};
            int_container expected = {0, 0, 10};
            int_container expected_2 = {0, 1, 10};
            EXPECT_TRUE(all(equal(maximum(a, b), expected)));
            EXPECT_TRUE(all(equal(maximum(arange(0, 3), b), expected_2)));
        }

        TEST_CASE_TEMPLATE("amax", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container_1d = xop_test::rebind_container_t<container_1d, int>;
            int_container_2d a = {{0, 0, 3}, {1, 2, 10}};
            EXPECT_EQ(10, amax(a)());
            int_container_1d e1 = {1, 2, 10};
            EXPECT_EQ(e1, amax(a, {0}));
            int_container_1d e2 = {3, 10};
            EXPECT_EQ(e2, amax(a, {1}));
        }

        TEST_CASE_TEMPLATE("amin", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container_1d = xop_test::rebind_container_t<container_1d, int>;
            int_container_2d a = {{0, 0, 3}, {1, 2, 10}};
            EXPECT_EQ(0, amin(a)());
            int_container_1d e1 = {0, 0, 3};
            EXPECT_EQ(e1, amin(a, {0}));
            int_container_1d e2 = {0, 1};
            EXPECT_EQ(e2, amin(a, {1}));
        }

        TEST_CASE_TEMPLATE("all", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container_1d = xop_test::rebind_container_t<container_1d, int>;
            int_container_1d a = {1, 1, 3};
            EXPECT_EQ(true, all(a));
            int_container_2d b = {{0, 2, 1}, {2, 1, 0}};
            EXPECT_EQ(false, all(b));
        }

        TEST_CASE_TEMPLATE("all_layout", TypeParam, XOPERATION_TEST_TYPES)
        {
            xarray<int, layout_type::row_major> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            xarray<int, layout_type::column_major> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            EXPECT_EQ(a(0, 1), b(0, 1));
            EXPECT_TRUE(all(equal(a, b)));
        }

        TEST_CASE_TEMPLATE("nonzero", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container_1d = xop_test::rebind_container_t<container_1d, int>;
            using container_3d = redim_container_t<TypeParam, 3>;
            using bool_container = xop_test::rebind_container_t<container_3d, bool>;
            using shape_type = typename container_3d::shape_type;

            int_container_1d a = {1, 0, 3};
            std::vector<std::vector<std::size_t>> expected = {{0, 2}};
            EXPECT_EQ(expected, nonzero(a));

            int_container_2d b = {{0, 2, 1}, {2, 1, 0}};
            std::vector<std::vector<std::size_t>> expected_b = {{0, 0, 1, 1}, {1, 2, 0, 1}};
            EXPECT_EQ(expected_b, nonzero(b));

            auto c = equal(b, 0);
            std::vector<std::vector<std::size_t>> expected_c = {{0, 1}, {0, 2}};
            EXPECT_EQ(expected_c, nonzero(c));

            shape_type s = {3, 3, 3};
            bool_container d(s);
            std::fill(d.begin(), d.end(), true);

            auto d_nz = nonzero(d);
            EXPECT_EQ(size_t(3), d_nz.size());
            EXPECT_EQ(size_t(27 * 27 * 27), d_nz[0].size() * d_nz[1].size() * d_nz[2].size());
        }

        TEST_CASE_TEMPLATE("where_only_condition", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            int_container_2d a = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            std::vector<std::vector<std::size_t>> expected = {{0, 1, 2}, {0, 1, 2}};
            EXPECT_EQ(expected, where(a));
        }

        TEST_CASE_TEMPLATE("where", TypeParam, XOPERATION_TEST_TYPES)
        {
            TypeParam a = {{1, 2, 3}, {0, 1, 0}, {0, 4, 1}};
            double b = 1.0;
            TypeParam res = where(a > b, b, a);
            TypeParam expected = {{1, 1, 1}, {0, 1, 0}, {0, 1, 1}};
            EXPECT_EQ(expected, res);

#ifdef XTENSOR_USE_XSIMD
            // This will fail to compile if simd is broken for conditional_ternary
            auto func = where(a > b, b, a);
            auto s = func.template load_simd<xsimd::aligned_mode>(0);
            (void) s;

            using assign_traits = xassign_traits<TypeParam, decltype(func)>;

            EXPECT_TRUE(assign_traits::simd_linear_assign());
#endif
        }

        TEST_CASE_TEMPLATE("where_optional", TypeParam, XOPERATION_TEST_TYPES)
        {
            using opt_type = xoptional_assembly<TypeParam, xtensor<bool, 2>>;
            auto missing = xtl::missing<double>();
            opt_type a = {{1, missing, 3}, {0, 1, 0}, {missing, 4, 1}};
            double b = 1.0;

            opt_type res = where(a > b, b, a);
            opt_type expected = {{1, missing, 1}, {0, 1, 0}, {missing, 1, 1}};
            EXPECT_EQ(expected, res);

            opt_type res1 = where(true, a + 3, a);
            opt_type expected1 = {{4, missing, 6}, {3, 4, 3}, {missing, 7, 4}};
            EXPECT_EQ(expected1, res1);
        }

        TEST_CASE_TEMPLATE("where_cast", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            int_container_2d a = {{0, 1, 0}, {3, 0, 5}};
            double res1 = 1.2;
            TypeParam b = where(equal(a, 0.0), res1, 0.0);
            TypeParam expected = {{1.2, 0., 1.2}, {0., 1.2, 0.}};
            EXPECT_EQ(b, expected);
        }

        TEST_CASE_TEMPLATE("argwhere", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_2d = xop_test::rebind_container_t<TypeParam, int>;
            using container_1d = redim_container_t<TypeParam, 1>;
            using int_container_1d = xop_test::rebind_container_t<container_1d, int>;
            using container_3d = redim_container_t<TypeParam, 3>;
            using bool_container = xop_test::rebind_container_t<container_3d, bool>;
            using shape_type = typename container_3d::shape_type;

            int_container_1d a = {1, 0, 3};
            std::vector<xindex_type_t<typename int_container_1d::shape_type>> expected = {{0}, {2}};
            EXPECT_EQ(expected, argwhere(a));

            int_container_2d b = {{0, 2, 1}, {2, 1, 0}};
            std::vector<xindex_type_t<typename int_container_2d::shape_type>> expected_b = {
                {0, 1},
                {0, 2},
                {1, 0},
                {1, 1}};
            EXPECT_EQ(expected_b, argwhere(b));

            auto c = equal(b, 0);
            std::vector<xindex_type_t<typename int_container_2d::shape_type>> expected_c = {{0, 0}, {1, 2}};
            EXPECT_EQ(expected_c, argwhere(c));

            shape_type s = {3, 3, 3};
            bool_container d(s);
            std::fill(d.begin(), d.end(), true);

            auto d_nz = argwhere(d);
            EXPECT_EQ(size_t(3 * 3 * 3), d_nz.size());
            xindex_type_t<typename container_3d::shape_type> last_idx = {2, 2, 2};
            EXPECT_EQ(last_idx, d_nz.back());
        }

        TEST_CASE_TEMPLATE("cast", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_t = xop_test::rebind_container_t<TypeParam, int>;
            using shape_type = typename int_container_t::shape_type;
            shape_type shape = {3, 2};
            int_container_t a(shape, 5);
            auto ref = static_cast<double>(a(0, 0)) / 2;
            auto actual = (cast<double>(a) / 2)(0, 0);
            EXPECT_EQ(ref, actual);
        }

        TEST_CASE_TEMPLATE("cast_custom_type", TypeParam, XOPERATION_TEST_TYPES)
        {
            using vtype_container_t = xop_test::rebind_container_t<TypeParam, vtype>;
            using shape_type = typename vtype_container_t::shape_type;
            shape_type shape = {3, 2};
            vtype_container_t a(shape);
            auto ref = static_cast<double>(a(0, 0));
            auto actual = (cast<double>(a))(0, 0);
            EXPECT_EQ(ref, actual);
        }

        TEST_CASE_TEMPLATE("mixed_arithmetic", TypeParam, XOPERATION_TEST_TYPES)
        {
            using int_container_t = xop_test::rebind_container_t<TypeParam, int>;
            TypeParam a = {{0., 1., 2.}, {3., 4., 5.}};
            int_container_t b = {{0, 1, 2}, {3, 4, 5}};
            int_container_t c = b;
            TypeParam res = a + (b + c);
            TypeParam expected = {{0., 3., 6.}, {9., 12., 15.}};
            EXPECT_EQ(res, expected);
        }

        TEST_CASE_TEMPLATE("assign_traits", TypeParam, XOPERATION_TEST_TYPES)
        {
            TypeParam a = {{0., 1., 2.}, {3., 4., 5.}};
            TypeParam b = {{0., 1., 2.}, {3., 4., 5.}};

            SUBCASE("xarray<double> + xarray<double>")
            {
                auto fd = a + b;
                using assign_traits_double = xassign_traits<TypeParam, decltype(fd)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_double::simd_linear_assign());
#else
                // SFINAE on load_simd is broken on mingw when xsimd is disabled. This using
                // triggers the same error as the one caught by mingw.
                using return_type = decltype(fd.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_double::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("double * xarray<double>")
            {
                xscalar<double> sd = 2.;
                auto fsd = sd * a;
                using assign_traits_scalar_double = xassign_traits<TypeParam, decltype(fsd)>;
#if XTENSOR_USE_XSIMD
                auto batch = fsd.template load_simd<double>(0);
                (void) batch;
                EXPECT_TRUE(assign_traits_scalar_double::simd_linear_assign());
#else
                using return_type = decltype(fsd.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_scalar_double::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("xarray<double> + xarray<int>")
            {
                using int_container_t = xop_test::rebind_container_t<TypeParam, int>;
                int_container_t c = {{0, 1, 2}, {3, 4, 5}};
                auto fm = a + c;
                using assign_traits_mixed = xassign_traits<TypeParam, decltype(fm)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_mixed::simd_linear_assign());
#else
                using return_type = decltype(fm.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_mixed::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("int * xarray<double>")
            {
                xscalar<int> si = 2;
                auto fsm = si * a;
                using assign_traits_scalar_mixed = xassign_traits<TypeParam, decltype(fsm)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_scalar_mixed::simd_linear_assign());
#else
                using return_type = decltype(fsm.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_scalar_mixed::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("xarray<double> + xarray<char>")
            {
                using char_container_t = xop_test::rebind_container_t<TypeParam, char>;
                char_container_t d = {{0, 1, 2}, {3, 4, 5}};
                auto fdc = a + d;
                using assign_traits_char_double = xassign_traits<TypeParam, decltype(fdc)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_char_double::simd_linear_assign());
#else
                using return_type = decltype(fdc.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_char_double::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("xarray<double> + xarray<my_double>")
            {
                using md_container_t = xop_test::rebind_container_t<TypeParam, my_double>;
                md_container_t d = {{0, 1, 2}, {3, 4, 5}};
                auto fdm = a + d;
                using assign_traits_md_double = xassign_traits<TypeParam, decltype(fdm)>;
#if XTENSOR_USE_XSIMD
                EXPECT_FALSE(assign_traits_md_double::simd_linear_assign());
#else
                using return_type = decltype(fdm.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_md_double::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
            }

            SUBCASE("xarray<double> > xarray<double>")
            {
                auto fgt = a > b;
                using bool_container_t = xop_test::rebind_container_t<TypeParam, bool>;
                using assign_traits_gt = xassign_traits<bool_container_t, decltype(fgt)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_gt::simd_linear_assign());
#else
                using return_type = decltype(fgt.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_gt::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, bool>::value));
#endif
            }

            SUBCASE("xarray<bool> || xarray<bool>")
            {
                using bool_container_t = xop_test::rebind_container_t<TypeParam, bool>;
                bool_container_t b0 = {{true, false, true}, {false, false, true}};
                bool_container_t b1 = {{true, true, false}, {false, true, true}};
                auto fb = b0 || b1;
                using assign_traits_bool_bool = xassign_traits<bool_container_t, decltype(fb)>;
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(assign_traits_bool_bool::simd_linear_assign());
#else
                using return_type = decltype(fb.template load_simd<aligned_mode>(std::size_t(0)));
                EXPECT_FALSE(assign_traits_bool_bool::simd_linear_assign());
                EXPECT_TRUE((std::is_same<return_type, bool>::value));
#endif
            }
        }


        TEST(operation, mixed_assign)
        {
            xt::xarray<double> asrc = {1., 2.};
            xt::xarray<std::size_t> bsrc = {std::size_t(3), std::size_t(4)};

            xt::xarray<double> a(asrc);
            xt::xarray<double> aexp = {3., 4.};
            a = bsrc;

            xt::xarray<std::size_t> b(bsrc);
            xt::xarray<std::size_t> bexp = {std::size_t(1), std::size_t(2)};
            b = asrc;
            EXPECT_EQ(b, bexp);
        }

        TEST(operation, mixed_bool_assign)
        {
            xt::xarray<double> a = {1., 6.};
            xt::xarray<double> b = {2., 3.};
            using uchar = unsigned char;
            xt::xarray<uchar> res = a > b;
            xt::xarray<uchar> exp = {uchar(0), uchar(1)};
            EXPECT_EQ(res, exp);
        }

        TEST(operation, dynamic_simd_assign)
        {
            using array_type = xt::xarray<double, layout_type::dynamic>;
            array_type a({2, 3}, layout_type::row_major);
            array_type b({2, 3}, layout_type::column_major);

            auto frr = a + a;
            auto frc = a + b;
            auto fcc = b + b;

            using frr_traits = xassign_traits<array_type, decltype(frr)>;
            using frc_traits = xassign_traits<array_type, decltype(frc)>;
            using fcc_traits = xassign_traits<array_type, decltype(fcc)>;

            EXPECT_FALSE(frr_traits::simd_linear_assign());
            EXPECT_FALSE(frc_traits::simd_linear_assign());
            EXPECT_FALSE(fcc_traits::simd_linear_assign());

            SUBCASE("row_major + row_major")
            {
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(frr_traits::simd_linear_assign(a, frr));
#else
                EXPECT_FALSE(frr_traits::simd_linear_assign(a, frr));
#endif
                EXPECT_FALSE(frr_traits::simd_linear_assign(b, frr));
            }

            SUBCASE("row_major + column_major")
            {
                EXPECT_FALSE(frc_traits::simd_linear_assign(a, frc));
                EXPECT_FALSE(frc_traits::simd_linear_assign(b, frc));
            }

            SUBCASE("row_major + column_major")
            {
                EXPECT_FALSE(fcc_traits::simd_linear_assign(a, fcc));
#if XTENSOR_USE_XSIMD
                EXPECT_TRUE(fcc_traits::simd_linear_assign(b, fcc));
#else
                EXPECT_FALSE(fcc_traits::simd_linear_assign(b, fcc));
#endif
            }
        }

        TEST_CASE("left_shift")
        {
            xarray<int> arr({5, 1, 1000});
            xarray<int> arr2({2, 1, 3});
            xarray<int> res1 = left_shift(arr, 4);
            xarray<int> res2 = left_shift(arr, arr2);
            EXPECT_EQ(left_shift(arr, 4)(1), 16);
            xarray<int> expected1 = {80, 16, 16000};
            xarray<int> expected2 = {20, 2, 8000};

            EXPECT_EQ(expected1, res1);
            EXPECT_EQ(expected2, res2);

            xarray<int> res3 = arr << 4;
            xarray<int> res4 = arr << arr2;
            EXPECT_EQ(expected1, res3);
            EXPECT_EQ(expected2, res4);
        }

        TEST_CASE("right_shift")
        {
            xarray<int> arr({5, 1, 1000});
            xarray<int> arr2({2, 1, 3});
            xarray<int> res1 = right_shift(arr, 4);
            xarray<int> res2 = right_shift(arr, arr2);
            EXPECT_EQ(right_shift(arr, 4)(1), 0);
            xarray<int> expected1 = {0, 0, 62};
            xarray<int> expected2 = {1, 0, 125};

            EXPECT_EQ(expected1, res1);
            EXPECT_EQ(expected2, res2);

            xarray<int> res3 = arr >> 4;
            xarray<int> res4 = arr >> arr2;
            EXPECT_EQ(expected1, res3);
            EXPECT_EQ(expected2, res4);
        }
    }

#undef XOPERATION_TEST_TYPES

}
