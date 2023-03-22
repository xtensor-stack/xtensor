/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <type_traits>
#include <vector>

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xassign.hpp"
#include "xtensor/xlayout.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    TEST(xassign_strided, mix_shape_types)
    {
        auto check_linear_assign = [](auto a, auto b)
        {
            assert_compatible_shape(a, b);
            return xassign_traits<decltype(a), decltype(b)>::linear_assign(a, b, true);
        };
        auto check_strided_assign = [](auto a, auto b)
        {
            assert_compatible_shape(a, b);
#ifndef _WIN32
            static_assert(
                xassign_traits<decltype(a), decltype(b)>::strided_assign(),
                "Failed to do strided assign"
            );
#endif
            return strided_assign_detail::get_loop_sizes(a, b).can_do_strided_assign;
        };
        {
            size_t size = 50;
            const std::array<size_t, 3> shape = {size, size, size};
            xt::xtensor<double, 3> a(shape), b(shape);
            auto core = xt::range(1, size - 1);
            auto lhs = xt::view(b, core, core, core);
            auto rhs = 1.0 / 7.0
                       * (xt::view(a, core, core, core) + xt::view(a, core, xt::range(2, size), core)
                          + xt::view(a, core, xt::range(0, size - 2), core)
                          + xt::view(a, xt::range(2, size), core, core)
                          + xt::view(a, xt::range(0, size - 2), core, core));
            xt::noalias(lhs) = rhs;
            EXPECT_TRUE(check_strided_assign(lhs, rhs));
        }
        {
            auto data = std::vector<double>{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
            auto simple_xtensor_12 = xt::xtensor<double, 2, layout_type::row_major>{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}};
            auto simple_xtensor_16 = xt::xtensor<double, 2, layout_type::row_major>{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}};
            auto adapter_strided_noncont = xt::adapt(
                data.data(),
                12,
                xt::no_ownership(),
                std::vector<size_t>{4, 3},
                std::vector<size_t>{4, 1}
            );
            auto adapter_strided_cont = xt::adapt(
                data.data() + 1,
                12,
                xt::no_ownership(),
                std::vector<size_t>{3, 4},
                std::vector<size_t>{4, 1}
            );
            auto adapter_strided_noncont_43 = xt::adapt(
                data.data(),
                12,
                xt::no_ownership(),
                std::vector<size_t>{3, 4},
                std::vector<size_t>{5, 1}
            );
            EXPECT_FALSE(check_strided_assign(adapter_strided_noncont, xt::transpose(adapter_strided_cont)));
            auto contiguous_view = xt::view(simple_xtensor_12, xt::range(0, 3), xt::all());
            EXPECT_TRUE(check_linear_assign(contiguous_view, adapter_strided_cont));
            EXPECT_FALSE(check_linear_assign(contiguous_view, xt::transpose(adapter_strided_noncont)));
            EXPECT_FALSE(check_linear_assign(xt::transpose(adapter_strided_noncont), contiguous_view));
            EXPECT_FALSE(check_linear_assign(contiguous_view, adapter_strided_noncont_43));
            EXPECT_FALSE(check_linear_assign(adapter_strided_noncont_43, contiguous_view));
            EXPECT_TRUE(check_strided_assign(contiguous_view, adapter_strided_noncont_43));
            EXPECT_TRUE(check_strided_assign(adapter_strided_noncont_43, contiguous_view));
            auto contiguous_view2 = xt::view(simple_xtensor_12, xt::range(1, 3), xt::all());
            auto contiguous_view3 = xt::view(simple_xtensor_12, xt::range(0, 2), xt::all());
            EXPECT_TRUE(check_linear_assign(contiguous_view2, contiguous_view3));
            {
                auto view_noncont = xt::view(simple_xtensor_16, xt::all(), xt::range(0, 3));
                EXPECT_FALSE(check_linear_assign(adapter_strided_noncont, view_noncont));
                EXPECT_TRUE(check_strided_assign(adapter_strided_noncont, view_noncont));
            }
            {
                auto view_cont = xt::view(simple_xtensor_16, xt::range(0, 1), xt::range(0, 4));
                auto view_cont2 = xt::view(simple_xtensor_16, xt::range(3, 4), xt::range(0, 4));
                EXPECT_TRUE(check_linear_assign(view_cont, view_cont2));
                auto view_noncont = xt::view(simple_xtensor_16, xt::range(0, 4), xt::range(0, 3));
                auto view_noncont2 = xt::view(simple_xtensor_16, xt::range(0, 4), xt::range(1, 4));
                EXPECT_FALSE(check_linear_assign(view_noncont, view_noncont2));
                EXPECT_TRUE(check_strided_assign(view_noncont, view_noncont2));
            }

            {
                std::vector<double> data2{-1, -1, -1, -1};
                auto linear_adapter = xt::adapt<layout_type::row_major>(
                    data2.data(),
                    4,
                    xt::no_ownership(),
                    std::vector<size_t>{4, 1}
                );
                auto adapter_cont2 = xt::adapt<layout_type::row_major>(
                    data.data(),
                    4,
                    xt::no_ownership(),
                    std::vector<size_t>{1, 4}
                );
                EXPECT_TRUE(linear_adapter.is_contiguous());
                EXPECT_TRUE(adapter_cont2.is_contiguous());
                bool success_one = check_linear_assign(linear_adapter, xt::transpose(adapter_cont2));
                bool success_two = check_linear_assign(adapter_cont2, xt::transpose(linear_adapter));
                EXPECT_TRUE(success_one && success_two);
                auto adapter_noncont_singlecol = xt::adapt(
                    data.data(),
                    4,
                    xt::no_ownership(),
                    std::vector<size_t>{4, 1},
                    std::vector<size_t>{4, 1}
                );
                EXPECT_FALSE(check_linear_assign(adapter_noncont_singlecol, linear_adapter));
                EXPECT_FALSE(check_linear_assign(linear_adapter, adapter_noncont_singlecol));
                EXPECT_FALSE(check_strided_assign(adapter_noncont_singlecol, linear_adapter));
                EXPECT_FALSE(check_strided_assign(linear_adapter, adapter_noncont_singlecol));
                auto adapter_noncont_twocol = xt::adapt(
                    data.data(),
                    4,
                    xt::no_ownership(),
                    std::vector<size_t>{2, 2},
                    std::vector<size_t>{4, 1}
                );
                EXPECT_FALSE(check_linear_assign(adapter_noncont_twocol, linear_adapter.reshape({2, 2})));
                EXPECT_FALSE(check_linear_assign(linear_adapter.reshape({2, 2}), adapter_noncont_twocol));
                EXPECT_TRUE(check_strided_assign(adapter_noncont_twocol, linear_adapter.reshape({2, 2})));
                EXPECT_TRUE(check_strided_assign(linear_adapter.reshape({2, 2}), adapter_noncont_twocol));
                auto adapter_zero_strides = xt::adapt(
                    data.data(),
                    4,
                    xt::no_ownership(),
                    std::vector<size_t>{2, 2},
                    std::vector<size_t>{0, 0}
                );
                EXPECT_FALSE(check_linear_assign(adapter_zero_strides, adapter_zero_strides));
                EXPECT_FALSE(check_strided_assign(adapter_zero_strides, adapter_zero_strides));
                EXPECT_FALSE(check_linear_assign(adapter_zero_strides, linear_adapter));
            }

            {
                std::vector<double> data2{-1, -1, -1, -1, -1, -1};
                auto linear_adapter2 = xt::adapt<layout_type::row_major>(
                    data2.data(),
                    6,
                    xt::no_ownership(),
                    std::vector<size_t>{3, 2}
                );
                auto strided_adapter = xt::adapt(
                    data.data(),
                    6,
                    xt::no_ownership(),
                    std::vector<size_t>{3, 2},
                    std::vector<size_t>{4, 1}
                );
                EXPECT_FALSE(check_linear_assign(linear_adapter2, strided_adapter));
                EXPECT_FALSE(check_linear_assign(strided_adapter, linear_adapter2));
                EXPECT_TRUE(check_strided_assign(linear_adapter2, strided_adapter));
                EXPECT_TRUE(check_strided_assign(strided_adapter, linear_adapter2));
                xt::noalias(strided_adapter) = linear_adapter2;
                auto result_expected = std::vector<double>{
                    -1, -1, 3, 4, -1, -1, 7, 8, -1, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
                EXPECT_EQ(data, result_expected);
            }
        }
    }
}
