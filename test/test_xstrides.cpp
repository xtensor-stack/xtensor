/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xtensor/xarray.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    TEST(xstrides, broadcast_shape)
    {
        using shape_type = std::vector<std::size_t>;
        shape_type s1 = {2, 4, 3};
        shape_type s2 = {2, 1, 3};
        shape_type s3 = {2, 0, 3};
        shape_type s4 = uninitialized_shape<shape_type>(3);

        shape_type s5 = s2;
        bool t1 = broadcast_shape(s1, s5);
        EXPECT_EQ(s5, s1);
        EXPECT_FALSE(t1);

        shape_type s6 = s2;
        bool t2 = broadcast_shape(s3, s6);
        EXPECT_EQ(s6, s3);
        EXPECT_FALSE(t2);

        shape_type s7 = s3;
        XT_EXPECT_ANY_THROW(broadcast_shape(s1, s7));

        shape_type s8 = s4;
        bool t3 = broadcast_shape(s1, s8);
        EXPECT_EQ(s8, s1);
        EXPECT_TRUE(t3);

        shape_type s9 = s4;
        bool t4 = broadcast_shape(s2, s9);
        EXPECT_EQ(s9, s2);
        EXPECT_TRUE(t4);

        shape_type s10 = s4;
        bool t5 = broadcast_shape(s3, s10);
        EXPECT_EQ(s10, s3);
        EXPECT_TRUE(t5);
    }

    TEST(xstrides, free_function_2d_row_major)
    {
        xt::xarray<int, xt::layout_type::row_major> a = xt::ones<int>({1, 3});
        using stype = std::vector<std::ptrdiff_t>;
        std::ptrdiff_t sof = sizeof(int);

        EXPECT_EQ(xt::strides(a), stype({3, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::normal), stype({3, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::internal), stype({0, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::bytes), stype({3 * sof, sof}));

        EXPECT_TRUE(xt::strides(a, 0) == 3);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::normal) == 3);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::internal) == 0);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::bytes) == 3 * sof);

        EXPECT_TRUE(xt::strides(a, 1) == 1);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::normal) == 1);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::internal) == 1);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::bytes) == sof);
    }

    TEST(xstrides, free_function_4d_row_major)
    {
        xt::xarray<int, xt::layout_type::row_major> a = xt::ones<int>({5, 4, 1, 4});
        using stype = std::vector<std::ptrdiff_t>;
        std::ptrdiff_t sof = sizeof(int);

        EXPECT_EQ(xt::strides(a), stype({16, 4, 4, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::normal), stype({16, 4, 4, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::internal), stype({16, 4, 0, 1}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::bytes), stype({16 * sof, 4 * sof, 4 * sof, 1 * sof}));

        EXPECT_TRUE(xt::strides(a, 0) == 16);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::normal) == 16);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::internal) == 16);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::bytes) == 16 * sof);

        EXPECT_TRUE(xt::strides(a, 1) == 4);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::normal) == 4);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::internal) == 4);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::bytes) == 4 * sof);

        EXPECT_TRUE(xt::strides(a, 2) == 4);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::normal) == 4);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::internal) == 0);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::bytes) == 4 * sof);

        EXPECT_TRUE(xt::strides(a, 3) == 1);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::normal) == 1);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::internal) == 1);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::bytes) == sof);
    }

    TEST(xstrides, free_function_4d_column_major)
    {
        xt::xarray<int, xt::layout_type::column_major> a = xt::ones<int>({5, 4, 1, 4});
        using stype = std::vector<std::ptrdiff_t>;
        std::ptrdiff_t sof = sizeof(int);

        EXPECT_EQ(xt::strides(a), stype({1, 5, 20, 20}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::normal), stype({1, 5, 20, 20}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::internal), stype({1, 5, 0, 20}));
        EXPECT_EQ(xt::strides(a, xt::stride_type::bytes), stype({sof, 5 * sof, 20 * sof, 20 * sof}));

        EXPECT_TRUE(xt::strides(a, 0) == 1);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::normal) == 1);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::internal) == 1);
        EXPECT_TRUE(xt::strides(a, 0, xt::stride_type::bytes) == sof);

        EXPECT_TRUE(xt::strides(a, 1) == 5);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::normal) == 5);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::internal) == 5);
        EXPECT_TRUE(xt::strides(a, 1, xt::stride_type::bytes) == 5 * sof);

        EXPECT_TRUE(xt::strides(a, 2) == 20);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::normal) == 20);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::internal) == 0);
        EXPECT_TRUE(xt::strides(a, 2, xt::stride_type::bytes) == 20 * sof);

        EXPECT_TRUE(xt::strides(a, 3) == 20);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::normal) == 20);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::internal) == 20);
        EXPECT_TRUE(xt::strides(a, 3, xt::stride_type::bytes) == 20 * sof);
    }

    TEST(xstrides, unravel_from_strides)
    {
        SUBCASE("row_major strides")
        {
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 1, 1};
            auto offset = element_offset<std::ptrdiff_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, rm.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        SUBCASE("column_major strides")
        {
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 1, 1};
            auto offset = element_offset<std::ptrdiff_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, cm.strides(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        SUBCASE("unit_major strides")
        {
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 0, 1};
            auto offset = element_offset<std::ptrdiff_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, um.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }

    TEST(xstrides, unravel_index)
    {
        SUBCASE("row_major strides")
        {
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 1, 1};
            auto offset = element_offset<std::size_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, rm.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        SUBCASE("column_major strides")
        {
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 1, 1};
            auto offset = element_offset<std::size_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, cm.shape(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        SUBCASE("unit_major strides")
        {
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = {2, 0, 1};
            auto offset = element_offset<std::size_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, um.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }

    TEST(xstrides, do_match_strides)
    {
        using vector_type = std::vector<std::size_t>;
        vector_type shape_0 = {2, 1, 4};

        vector_type strides_0 = {4, 0, 1};
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_0, xt::layout_type::row_major, true));

        vector_type strides_1 = {4, 4, 1};
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_1, xt::layout_type::row_major, false));

        vector_type strides_2 = {1, 0, 2};
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_2, xt::layout_type::column_major, true));

        vector_type strides_3 = {1, 2, 2};
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_3, xt::layout_type::column_major, false));

        vector_type shape_1 = {2, 1, 2, 4};
        vector_type strides_4 = {8, 8, 4, 1};
        EXPECT_TRUE(xt::do_strides_match(shape_1, strides_4, xt::layout_type::row_major, false));

        vector_type strides_5 = {8, 1, 8, 1};
        EXPECT_FALSE(xt::do_strides_match(shape_1, strides_5, xt::layout_type::row_major, false));

        vector_type shape_2 = {2, 2, 1, 4};
        vector_type strides_6 = {1, 2, 4, 4};
        EXPECT_TRUE(xt::do_strides_match(shape_2, strides_6, xt::layout_type::column_major, false));

        vector_type strides_7 = {1, 2, 1, 4};
        EXPECT_FALSE(xt::do_strides_match(shape_2, strides_7, xt::layout_type::column_major, false));
    }

    TEST(xstrides, ravel_index)
    {
        xt::uvector<std::size_t> index = {1, 1, 1};
        xt::uvector<std::size_t> shape = {10, 20, 30};
        auto idx = xt::ravel_index(index, shape, xt::layout_type::row_major);
        EXPECT_EQ(idx, 631);
    }
}
