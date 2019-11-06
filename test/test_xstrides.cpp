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
#include "test_common.hpp"

namespace xt
{
    TEST(xstrides, broadcast_shape)
    {
        using shape_type = std::vector<std::size_t>;
        shape_type s1 = { 2, 4, 3 };
        shape_type s2 = { 2, 1, 3 };
        shape_type s3 = { 2, 0, 3 };
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

    TEST(xstrides, unravel_from_strides)
    {
        {
            SCOPED_TRACE("row_major strides");
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::ptrdiff_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, rm.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("column_major strides");
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::ptrdiff_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, cm.strides(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("unit_major strides");
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 0, 1 };
            auto offset = element_offset<std::ptrdiff_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, um.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }

    TEST(xstrides, unravel_index)
    {
        {
            SCOPED_TRACE("row_major strides");
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, rm.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("column_major strides");
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, cm.shape(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("unit_major strides");
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::ptrdiff_t>;
            index_type index = { 2, 0, 1 };
            auto offset = element_offset<std::size_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, um.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }

    TEST(xstrides, do_match_strides)
    {
        using vector_type = std::vector<std::size_t>;
        vector_type shape_0 = { 2, 1, 4 };

        vector_type strides_0 = { 4, 0, 1 };
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_0, xt::layout_type::row_major, true));

        vector_type strides_1 = { 4, 4, 1 };
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_1, xt::layout_type::row_major, false));

        vector_type strides_2 = { 1, 0, 2 };
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_2, xt::layout_type::column_major, true));

        vector_type strides_3 = { 1, 2, 2 };
        EXPECT_TRUE(xt::do_strides_match(shape_0, strides_3, xt::layout_type::column_major, false));

        vector_type shape_1 = { 2, 1, 2, 4 };
        vector_type strides_4 = { 8, 8, 4, 1 };
        EXPECT_TRUE(xt::do_strides_match(shape_1, strides_4, xt::layout_type::row_major, false));

        vector_type strides_5 = { 8, 1, 8, 1 };
        EXPECT_FALSE(xt::do_strides_match(shape_1, strides_5, xt::layout_type::row_major, false));

        vector_type shape_2 = { 2, 2, 1, 4 };
        vector_type strides_6 = { 1, 2, 4, 4 };
        EXPECT_TRUE(xt::do_strides_match(shape_2, strides_6, xt::layout_type::column_major, false));

        vector_type strides_7 = { 1, 2, 1, 4 };
        EXPECT_FALSE(xt::do_strides_match(shape_2, strides_7, xt::layout_type::column_major, false));
    }
}
