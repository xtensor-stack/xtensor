/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "test_common.hpp"

namespace xt
{
    TEST(xstrides, unravel_from_strides)
    {
        {
            SCOPED_TRACE("row_major strides");
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, rm.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("column_major strides");
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, cm.strides(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("unit_major strides");
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 0, 1 };
            auto offset = element_offset<std::size_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_from_strides(offset, um.strides(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }

    TEST(xstrides, unravel_index)
    {
        {
            SCOPED_TRACE("row_major strides");
            row_major_result<> rm;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(rm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, rm.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("column_major strides");
            column_major_result<> cm;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 1, 1 };
            auto offset = element_offset<std::size_t>(cm.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, cm.shape(), layout_type::column_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }

        {
            SCOPED_TRACE("unit_major strides");
            unit_shape_result<> um;
            using index_type = xt::dynamic_shape<std::size_t>;
            index_type index = { 2, 0, 1 };
            auto offset = element_offset<std::size_t>(um.strides(), index.cbegin(), index.cend());
            index_type unrav_index = unravel_index(offset, um.shape(), layout_type::row_major);
            EXPECT_TRUE(std::equal(unrav_index.cbegin(), unrav_index.cend(), index.cbegin()));
        }
    }
}